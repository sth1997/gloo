/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "gloo/allreduce.h"

#include <array>
#include <algorithm>
#include <cstring>

#include "gloo/common/logging.h"
#include "gloo/math.h"
#include "gloo/types.h"

#ifdef BREAKDOWN_ANALYSIS
#include <time.h>
#include <sys/time.h>

double get_wall_time()
{
  struct timeval time;
  if (gettimeofday(&time,NULL)){
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}
static double compressTime = 0.0;
static double decompressTime = 0.0;
static double addSparseTime = 0.0;
static double commTime = 0.0;
static double selectTime = 0.0;
#endif

static bool useMultiThread = true;

namespace gloo {

namespace {

struct CompressFormat
{
    unsigned int index;
    float value;
};

void to_sparse(float* const ptr, const size_t num, const size_t totalBytes)
{
    size_t nonzeroNum = num / 1000;
    memset(ptr + nonzeroNum, 0, totalBytes - nonzeroNum * sizeof(float));
}

/*
 * Compress format: ((unsigned int) index, (datatype) val)
 * Example: (0,0,0,3,0,0,1,0,0,0,0,0,7,0,0) -> ((3,3),(6,1),(12,7))
 */
void compress(const void* const src, void* const dst, const float topKVal, size_t count, size_t nonzeroCount)
{
    #ifdef BREAKDOWN_ANALYSIS
    double start = get_wall_time();
    #endif

    const float* const srcValue = (const float* const) src;
    CompressFormat* compressed = (CompressFormat*) dst;
    size_t compressNum = 0;
    if(count < 512 * 1024 || !useMultiThread)
    {
        for (int index = 0; index < count; ++index)
        if (srcValue[index] >= topKVal)
        {
            compressed[compressNum].index = (unsigned int) index;
            compressed[compressNum].value = srcValue[index];
            if (++compressNum == nonzeroCount)
                break;
        }
    }
    else        
    //chw multi-thread
    {
        int threadNum;
        if(count < 32 * 1024 * 1024)
            threadNum = 8;
        else
            threadNum = 16;
        static unsigned int** compressIdx_thread;
        compressIdx_thread = (unsigned int**)malloc(threadNum * sizeof(unsigned int*));
        for(int i = 0; i < threadNum; ++i)
            compressIdx_thread[i] = (unsigned int*)malloc(count / threadNum * sizeof(unsigned int));
        int* threadIdx = (int*)malloc(threadNum * sizeof(int));
        #pragma omp parallel num_threads(threadNum)
        {
            int rank = omp_get_thread_num() + 1;
            int size = omp_get_num_threads();
            int thread_index = 0;
            int endIdx = rank * (count / size);
            if (rank == size)
                endIdx = count;
            float compressKValLocal = topKVal;
            unsigned int* compressIdxLocal = compressIdx_thread[rank - 1];
            for(int i = (rank - 1) * (count /size); i < endIdx; ++i)
            {
                if(srcValue[i] >= compressKValLocal)
                    compressIdxLocal[thread_index++] = i;
            }  
            threadIdx[rank - 1] = thread_index;
        }
        //merge
        //int index = 0;
        //int compressNum = 0;
        for(int i = 0; i < threadNum; ++i)
        {
            unsigned int* compressIdx = compressIdx_thread[i];
            for(int j = 0; j < threadIdx[i]; ++j)
            {
                compressed[compressNum].index = compressIdx[j];
                compressed[compressNum].value = srcValue[compressIdx[j]];
                if(++compressNum == nonzeroCount)
                    break;
            }
            if(compressNum == nonzeroCount)
                break;
        }
    }
    GLOO_ENFORCE_EQ(compressNum, nonzeroCount);
    #ifdef BREAKDOWN_ANALYSIS
    double end = get_wall_time();
    compressTime = end - start;
    #endif
}

void multi_thread_memset(void* dst, int val, size_t size)
{
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    size_t size_per_thread = size / thread_count;
    char* startAddr = ((char*) dst) + size_per_thread * my_rank;
    if (my_rank != thread_count - 1)
        memset(startAddr, val, size_per_thread);
    else
    {
        size_t tmpSize = size - size_per_thread * my_rank;
        memset(startAddr, val, tmpSize);
    }
}

void decompress(const void* const src, void* const dst, size_t count, size_t nonzeroCount)
{
    #ifdef BREAKDOWN_ANALYSIS
    double start = get_wall_time();
    #endif
    
    // TODO : use multi-thread
    if(count < 8 * 1024 * 1024 || !useMultiThread)
        memset(dst, 0, sizeof(float) * count);
    else	
    {
        int threadNum;
        if(count < 32 * 1024 * 1024)
            threadNum = 2;
        else if(count < 128 * 1024 * 1024)
            threadNum = 4;
        else if(count <= 512 * 1024 * 1024)
            threadNum = 8;
        else
            threadNum = 16;
        #pragma omp parallel num_threads(threadNum)
        multi_thread_memset(dst, 0, sizeof(float) * count);
    }

    float* dstValue = (float*) dst;
    const CompressFormat* compressed = (CompressFormat*) src;
    // TODO : use multi-thread
    for (size_t i = 0; i < nonzeroCount; ++i)
        dstValue[compressed[i].index] = compressed[i].value;
    #ifdef BREAKDOWN_ANALYSIS
    double end = get_wall_time();
    decompressTime = end - start;
    #endif
}

/* Find the nearest power of 2 greater than this value*/
int nextPowerOfTwoGT(int value)
{
    int power2;
    for (power2 = 1; power2 <= value; power2 <<= 1) /* empty */;
    return power2;
}

void addSparse(const void* src1, const void* src2, void* dst, int nonzeroCount1, int nonzeroCount2, int& totalNonzeroCount)
{
    #ifdef BREAKDOWN_ANALYSIS
    double start = get_wall_time();
    #endif

    const CompressFormat* cp1 = (CompressFormat*) src1;
    const CompressFormat* cp2 = (CompressFormat*) src2;
    CompressFormat* cpAdd = (CompressFormat*) dst;
    int index1 = 0;
    int index2 = 0;
    totalNonzeroCount = nonzeroCount1 + nonzeroCount2;
    // TODO : use multi-thread
    int i;
    for (i = 0; i < nonzeroCount1 + nonzeroCount2; ++i)
        if (cp1[index1].index < cp2[index2].index)
        {
            cpAdd[i] = cp1[index1];
            if (++index1 == nonzeroCount1)
                break;
        }
        else if (cp1[index1].index > cp2[index2].index)
        {
            cpAdd[i] = cp2[index2];
            if (++index2 == nonzeroCount2)
                break;
        }
        else //srcIndex1[index1] == srcIndex2[index2]
        {
            cpAdd[i].index = cp1[index1].index;
            cpAdd[i].value = cp1[index1].value + cp2[index2].value;
            --totalNonzeroCount;
            ++index1;
            ++index2;
            if ((index1 == nonzeroCount1) || (index2 == nonzeroCount2))
                break;
        }
    
    ++i;
    /*
    while (index1 < nonzeroCount1 && index2 < nonzeroCount2)
    {
        if (cp1[index1].index != cp2[index2].index)
        {
            bool take1 = (cp1[index1].index < cp2[index2].index);
            cpAdd[i++] = take1 ? cp1[index1] : cp2[index2];
            index1 += take1;
            index2 += 1 - take1;
        }
        else
        {
            cpAdd[i].index = cp1[index1].index;
            cpAdd[i].value = cp1[index1++].value + cp2[index2++].value;
            --totalNonzeroCount;
        }
    }
    */

    if (index1 < nonzeroCount1)
        memcpy(&cpAdd[i], &cp1[index1], (sizeof(int) + sizeof(float)) * (nonzeroCount1 - index1));
    else if (index2 < nonzeroCount2)
        memcpy(&cpAdd[i], &cp2[index2], (sizeof(int) + sizeof(float)) * (nonzeroCount2 - index2));
    #ifdef BREAKDOWN_ANALYSIS
    double end = get_wall_time();
    addSparseTime += end - start;
    #endif
}

#define setNonzeroCountBuffer(buf, count) (*((size_t*)(buf->ptr)) = count)
#define getNonzeroCountBuffer(count, buf) (count = *((size_t*)(buf->ptr)))

void allreduceSparse(const gloo::Slot& slot, const std::shared_ptr<Context> context,
  AllreduceOptions& opts, transport::UnboundBuffer* sbuf,
  transport::UnboundBuffer* rbuf, transport::UnboundBuffer* tmp_buf, std::chrono::milliseconds& timeout,
  int size, int rank, size_t nonzeroCount, transport::UnboundBuffer*& resultBuf, size_t& retTotalNonzeroCount)
{
    transport::UnboundBuffer* tmpswap = NULL;
    transport::UnboundBuffer* tmpsend = sbuf;
    transport::UnboundBuffer* tmprecv = rbuf;
    transport::UnboundBuffer* tmpadd = tmp_buf;

    std::unique_ptr<uint8_t[]> tmpAllocation(new uint8_t[8]);
    std::unique_ptr<transport::UnboundBuffer> tmpBuffer =
      context->createUnboundBuffer(tmpAllocation.get(), 8);
    transport::UnboundBuffer* nonzeroCountBuffer = tmpBuffer.get();


    int newrank, newremote, extra_ranks, adjsize, remote, distance;

    adjsize = nextPowerOfTwoGT(size);
    /* Determine nearest power of two less than or equal to size */
    adjsize >>= 1;

    /* Handle non-power-of-two case:
       - Even ranks less than 2 * extra_ranks send their data to (rank + 1), and
       sets new rank to -1.
       - Odd ranks less than 2 * extra_ranks receive data from (rank - 1),
       apply appropriate operation, and set new rank to rank/2
       - Everyone else sets rank to rank - extra_ranks
    */
   extra_ranks = size - adjsize;
    size_t recvNonzeroCount, sendNonzeroCount = nonzeroCount;
    if (rank <  (2 * extra_ranks)) {
      if (0 == (rank & 1)) {
        // TODO : Combine these two MPI_Send.
        #ifdef BREAKDOWN_ANALYSIS
        double start = get_wall_time();
        #endif
        // TODO : combine these two send
        // TODO : move wait to somewhere
        GLOO_ENFORCE(
          context->getPair(rank + 1),
          "missing connection between rank " + std::to_string(rank) +
          " (this process) and rank " + std::to_string(rank + 1));
        setNonzeroCountBuffer(nonzeroCountBuffer, sendNonzeroCount);
        nonzeroCountBuffer->send(rank + 1, slot, 0, 8);
        nonzeroCountBuffer->waitSend(timeout);
        tmpsend->send(rank + 1, slot, 0, sendNonzeroCount * 2 * sizeof(float));
        tmpsend->waitSend(timeout);
        #ifdef BREAKDOWN_ANALYSIS
        double end = get_wall_time();
        commTime += end - start;
        #endif
        newrank = -1;
      } else {
        #ifdef BREAKDOWN_ANALYSIS
        double start = get_wall_time();
        #endif
        GLOO_ENFORCE(
          context->getPair(rank - 1),
          "missing connection between rank " + std::to_string(rank) +
          " (this process) and rank " + std::to_string(rank - 1));
        // TODO : combine these two recv
        nonzeroCountBuffer->recv(rank - 1, slot, 0, 8);
        nonzeroCountBuffer->waitRecv(timeout);
        getNonzeroCountBuffer(recvNonzeroCount, nonzeroCountBuffer);
        tmprecv->recv(rank - 1, slot, 0, recvNonzeroCount * 2 * sizeof(float));
        tmprecv->waitRecv(timeout);
        #ifdef BREAKDOWN_ANALYSIS
        double end = get_wall_time();
        commTime += end - start;
        #endif
        /* tmpsend = tmprecv (op) tmpsend */
        int totalNonzeroCount;
        addSparse(tmpsend->ptr, tmprecv->ptr, tmpadd->ptr, sendNonzeroCount,
                                    recvNonzeroCount, totalNonzeroCount);
        tmpswap = tmpadd;
        tmpadd = tmpsend;
        tmpsend = tmpswap;
        sendNonzeroCount = totalNonzeroCount;
        newrank = rank >> 1;
      }
    } else {
        newrank = rank - extra_ranks;
    }

    /* Communication/Computation loop
       - Exchange message with remote node.
       - Perform appropriate operation taking in account order of operations:
       result = value (op) result
    */
    for (distance = 0x1; distance < adjsize; distance <<= 1) {
        if (newrank < 0) break;
        /* Determine remote node */
        newremote = newrank ^ distance;
        remote = (newremote < extra_ranks)?
            (newremote * 2 + 1):(newremote + extra_ranks);
        
        #ifdef BREAKDOWN_ANALYSIS
        double start = get_wall_time();
        #endif
        
        GLOO_ENFORCE(
          context->getPair(remote),
          "missing connection between rank " + std::to_string(rank) +
          " (this process) and rank " + std::to_string(remote));
        /* Exchange the data */
        // TODO : combine send/recv and move wait
        if (rank < remote)
        {
          setNonzeroCountBuffer(nonzeroCountBuffer, sendNonzeroCount);
          nonzeroCountBuffer->send(remote, slot, 0, 8);
          nonzeroCountBuffer->waitSend(timeout);
          tmpsend->send(remote, slot, 0, sendNonzeroCount * 2 * sizeof(float));
          tmpsend->waitSend(timeout);
          nonzeroCountBuffer->recv(remote, slot, 0, 8);
          nonzeroCountBuffer->waitRecv(timeout);
          getNonzeroCountBuffer(recvNonzeroCount, nonzeroCountBuffer);
          tmprecv->recv(remote, slot, 0, recvNonzeroCount * 2 * sizeof(float));
          tmprecv->waitRecv(timeout);
        }
        else
        {
          nonzeroCountBuffer->recv(remote, slot, 0, 8);
          nonzeroCountBuffer->waitRecv(timeout);
          getNonzeroCountBuffer(recvNonzeroCount, nonzeroCountBuffer);
          tmprecv->recv(remote, slot, 0, recvNonzeroCount * 2 * sizeof(float));
          tmprecv->waitRecv(timeout);
          setNonzeroCountBuffer(nonzeroCountBuffer, sendNonzeroCount);
          nonzeroCountBuffer->send(remote, slot, 0, 8);
          nonzeroCountBuffer->waitSend(timeout);
          tmpsend->send(remote, slot, 0, sendNonzeroCount * 2 * sizeof(float));
          tmpsend->waitSend(timeout);
        }
        #ifdef BREAKDOWN_ANALYSIS
        double end = get_wall_time();
        commTime += end - start;
        #endif

        int totalNonzeroCount;
        addSparse(tmpsend->ptr, tmprecv->ptr, tmpadd->ptr, sendNonzeroCount,
                                    recvNonzeroCount, totalNonzeroCount);
        tmpswap = tmpadd;
        tmpadd = tmpsend;
        tmpsend = tmpswap;
        sendNonzeroCount = totalNonzeroCount;
        resultBuf = tmpsend;
    }

    /* Handle non-power-of-two case:
       - Odd ranks less than 2 * extra_ranks send result from tmpsend to
       (rank - 1)
       - Even ranks less than 2 * extra_ranks receive result from (rank + 1)
    */
    if (rank < (2 * extra_ranks)) {
        if (0 == (rank & 1)) {
            #ifdef BREAKDOWN_ANALYSIS
            double start = get_wall_time();
            #endif
            GLOO_ENFORCE(
              context->getPair(rank + 1),
              "missing connection between rank " + std::to_string(rank) +
              " (this process) and rank " + std::to_string(rank + 1));
            nonzeroCountBuffer->recv(rank + 1, slot, 0, 8);
            nonzeroCountBuffer->waitRecv(timeout);
            getNonzeroCountBuffer(recvNonzeroCount, nonzeroCountBuffer);
            tmpsend->recv(rank + 1, slot, 0, recvNonzeroCount * 2 * sizeof(float));
            tmpsend->waitRecv(timeout);
            #ifdef BREAKDOWN_ANALYSIS
            double end = get_wall_time();
            commTime += end - start;
            #endif
            resultBuf = tmpsend;
            retTotalNonzeroCount = recvNonzeroCount;
        } else {
            #ifdef BREAKDOWN_ANALYSIS
            double start = get_wall_time();
            #endif
            GLOO_ENFORCE(
              context->getPair(rank - 1),
              "missing connection between rank " + std::to_string(rank) +
              " (this process) and rank " + std::to_string(rank - 1));
            setNonzeroCountBuffer(nonzeroCountBuffer, sendNonzeroCount);
            nonzeroCountBuffer->send(rank - 1, slot, 0, 8);
            nonzeroCountBuffer->waitSend(timeout);
            tmpsend->send(rank - 1, slot, 0, sendNonzeroCount * 2 * sizeof(float));
            tmpsend->waitSend(timeout);
            #ifdef BREAKDOWN_ANALYSIS
            double end = get_wall_time();
            commTime += end - start;
            #endif
            resultBuf = tmpsend;
            retTotalNonzeroCount = sendNonzeroCount;
        }
    }
    retTotalNonzeroCount = sendNonzeroCount;
}

} // namespace

float randomSelect(float* buf, int count, int k)
{
  std::nth_element(buf, buf + (count - k - 1), buf + count);
  return buf[count - k - 1];
}

float select(const float* const buf, const int count)
{
    #ifdef BREAKDOWN_ANALYSIS
    double start = get_wall_time();
    #endif
    // TODO : try to merge some sample buffers into one buffer
    
    // sample buf[0~sampleCount-1]
    int sampleCount = count / 100;
    static int times = 0;
    ++times;
    static float* tmpBuf;
    static float** tmpbuf_thread;
    if (times == 1)
    {
        // NOTE sampleCount * 10 * sizeof(float). The code "tmpBuf[index++]" may cause some error if we just alloc sampleCount*sizeof(float).
        tmpBuf = (float*) mallocAlign(sampleCount * 10 * sizeof(float), 4);
    }
 
    float tmpKVal;
    bool sampleFailed = true;
    float ratio = 5.0f / 1000;
    int index = 0;
    while (sampleFailed)
    {
        memcpy(tmpBuf, buf, sampleCount * sizeof(float));
        tmpKVal = randomSelect(tmpBuf, sampleCount, sampleCount * ratio - 1);
        if(count <= 4 * 1024 * 1024 || !useMultiThread)
        {
          for (int i = 0; i < count; ++i)
            // do NOT set a[i]=0 if a[i] < tmpKVal
            if (buf[i] >= tmpKVal)
                tmpBuf[index++] = buf[i];
        }
        else		
        {
            int thread_count;
            if(count <= 32 * 1024 * 1024)
                thread_count = 4;
            else if(count <= 128 * 1024 * 1024)
                thread_count = 8;
            else
                thread_count = 16;
            tmpbuf_thread = (float**)malloc(thread_count * sizeof(float*));
            for(int i = 0; i < thread_count; ++i)
                tmpbuf_thread[i] = (float*)malloc(count / thread_count * sizeof(float));
            int* threadIdx = (int*) malloc(thread_count * sizeof(int));
            double start2 = get_wall_time();
            #pragma omp parallel num_threads(thread_count)
            {
                int rank = omp_get_thread_num() + 1;
                int size = omp_get_num_threads();
                int thread_index = 0;
                int endIdx = rank * (count / size);
                if (rank == size)
                    endIdx = count;
                float tmpKValLocal = tmpKVal;
                float* tmpBufLocal = tmpbuf_thread[rank - 1];
                for(int i = (rank - 1) * (count / size); i < endIdx; ++i)
                {
                    if(buf[i] >= tmpKValLocal)
                        tmpBufLocal[thread_index++] = buf[i];
                    //tmpbuf_thread[rank - 1][thread_index++] = buf[i];
                }
                // avoid false sharing
                threadIdx[rank - 1] = thread_index;
            }
            printf("parallelTime = %.5f\n", get_wall_time() - start2);
            //merge
            for(int i = 0; i < thread_count; ++i)
            {
                //for(int j = 0; tmpbuf_thread[i][j] != 0 && j < count / thread_count; j++)
                //    tmpBuf[index++] = tmpbuf_thread[i][j];
                memcpy(tmpBuf + index, tmpbuf_thread[i], threadIdx[i] * sizeof(float));
                index += threadIdx[i];
            }
            free(threadIdx);
        }
        printf("tmpKval = %.5f index = %d  count = %d\n", tmpKVal, index, count);
        if (index > sampleCount * 10)
        {
            printf("Index is too large! May cause buffer overflow error!");
            std::abort();
        }
        if (index < count / 1000)
        {
            sampleFailed = true;
            ratio *= 2.0f;
            continue;
        }
        float ret =  randomSelect(tmpBuf, index, count / 1000 - 1);
        #ifdef BREAKDOWN_ANALYSIS
        selectTime = get_wall_time() - start;
        #endif
        return ret;
    }
}


//Now, only support FLOAT and OP_SUM
void allreduce_sparse(AllreduceOptions& opts) {
  const auto& context = opts.context;
  std::vector<std::unique_ptr<transport::UnboundBuffer>>& in = opts.in;
  std::vector<std::unique_ptr<transport::UnboundBuffer>>& out = opts.out;
  const auto slot = Slot::build(kAllreduceSlotPrefix, opts.tag);

  // Sanity checks
  GLOO_ENFORCE_GT(out.size(), 0);
  GLOO_ENFORCE(opts.elements > 0);
  GLOO_ENFORCE(opts.elementSize > 0);
  GLOO_ENFORCE(opts.reduce != nullptr);

  // Assert the size of all inputs and outputs is identical.
  const size_t totalBytes = opts.elements * opts.elementSize;
  for (size_t i = 0; i < out.size(); i++) {
    GLOO_ENFORCE_EQ(out[i]->size, totalBytes);
  }
  for (size_t i = 0; i < in.size(); i++) {
    GLOO_ENFORCE_EQ(in[i]->size, totalBytes);
  }

  if (context->size == 1) {
    return;
  }

  GLOO_ENFORCE_EQ(out.size(), 1);
  GLOO_ENFORCE_EQ(in.size(), 0);

  #ifdef BREAKDOWN_ANALYSIS
  compressTime = 0.0;
  decompressTime = 0.0;
  addSparseTime = 0.0;
  commTime = 0.0;
  selectTime = 0.0;
  #endif

  // TODO : allocate a large buffer in progressGroupGloo to use everytime
  // TODO : select top-k
  size_t count = opts.elements;
  float topKVal = select((float*) out[0]->ptr, count);
  size_t nonzeroCount = opts.elements / 1000;

  int rank = context->rank;
  int size = context->size;

  // Allocate and initialize temporary send buffer
  // Now, we assume nonzeroCount in every process is the same.
  std::unique_ptr<uint8_t[]> tmpAllocation1(new uint8_t[(sizeof(int) + sizeof(float)) * nonzeroCount * size]);
  std::unique_ptr<uint8_t[]> tmpAllocation2(new uint8_t[(sizeof(int) + sizeof(float)) * nonzeroCount * size]);
  std::unique_ptr<transport::UnboundBuffer> tmpBuffer1 =
      context->createUnboundBuffer(tmpAllocation1.get(), (sizeof(int) + sizeof(float)) * nonzeroCount * size);
  std::unique_ptr<transport::UnboundBuffer> tmpBuffer2 =
      context->createUnboundBuffer(tmpAllocation2.get(), (sizeof(int) + sizeof(float)) * nonzeroCount * size);
  transport::UnboundBuffer* inplacebuf_free = tmpBuffer1.get();
  transport::UnboundBuffer* tmp_buf = tmpBuffer2.get();
  compress(out[0]->ptr, inplacebuf_free->ptr, topKVal, count, nonzeroCount);
  
  size_t totalNonzeroCount;
  transport::UnboundBuffer* resultBuf;
  allreduceSparse(slot, context, opts, inplacebuf_free, out[0].get(), tmp_buf, opts.timeout, size, rank, nonzeroCount, resultBuf, totalNonzeroCount);
  decompress(resultBuf->ptr, out[0]->ptr, count, totalNonzeroCount);

  #ifdef BREAKDOWN_ANALYSIS
  printf("selectTime = %.5f\n", selectTime);
  printf("compressTime = %.5f\n", compressTime);
  printf("decompressTime = %.5f\n", decompressTime);
  printf("addSparseTime = %.5f\n", addSparseTime);
  printf("commTime = %.5f\n", commTime);
  #endif
}

} // namespace gloo
