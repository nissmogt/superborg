/* Worker thread pool for data-parallel loops.
 *
 * pool_for() runs fn over [0, n_items) in chunks handed out by an atomic counter;
 * the calling thread works too. It returns once every item is done. It does not
 * wait for every worker to show up: on a loaded machine (phones, laptops running
 * other apps) a descheduled worker would otherwise stall each of the hundreds of
 * dispatches per token. Workers join a dispatch through a CAS on a word holding
 * (generation, closed flag, joined count), so a worker that wakes late cannot touch
 * a dispatch that has already finished.
 *
 * Idle workers spin briefly, then yield, then sleep on a condition variable.
 */
#include "internal.h"

#include <stdlib.h>

#if defined(_MSC_VER) && !defined(BORG_NO_THREADS)
#define BORG_NO_THREADS 1 /* MSVC has no pthreads: build with MinGW/clang for threads */
#endif

#if defined(BORG_NO_THREADS)

struct borg_pool {
    int n;
};

borg_pool *borg_pool_new(int n_threads) {
    (void)n_threads;
    borg_pool *p = (borg_pool *)calloc(1, sizeof(borg_pool));
    if (p) p->n = 1;
    return p;
}
void borg_pool_free(borg_pool *pool) { free(pool); }
int borg_pool_size(const borg_pool *pool) { return pool ? pool->n : 1; }
void pool_for(borg_pool *pool, int n_items, int chunk, borg_range_fn fn, void *ctx) {
    (void)pool;
    (void)chunk;
    if (n_items > 0) fn(ctx, 0, n_items, 0);
}

#else

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define cpu_relax() _mm_pause()
#elif defined(__aarch64__) || defined(__arm__)
#define cpu_relax() __asm__ __volatile__("yield" ::: "memory")
#else
#define cpu_relax() ((void)0)
#endif

#define SPIN_ITERS 20000   /* busy-wait before yielding (~1 ms) */
#define YIELD_ITERS 2000   /* sched_yield rounds before sleeping */

/* state word: generation in the high 32 bits, bit 31 = closed, low 31 bits = joined workers */
#define ST_GEN(s) ((uint32_t)((s) >> 32))
#define ST_CLOSED ((uint64_t)1 << 31)
#define ST_ACTIVE(s) ((uint32_t)((s) & 0x7FFFFFFFu))

typedef struct {
    borg_pool *pool;
    int ith;
} worker_arg;

struct borg_pool {
    int n; /* threads including the caller */
    pthread_t *threads;
    worker_arg *args;
    /* current dispatch */
    borg_range_fn fn;
    void *ctx;
    int n_items, chunk;
    _Alignas(64) atomic_int next;
    _Alignas(64) atomic_int done;
    _Alignas(64) _Atomic uint64_t state;
    _Alignas(64) atomic_int sleeping;
    atomic_int stop;
    pthread_mutex_t mu;
    pthread_cond_t cv;
};

static void run_chunks(borg_pool *p, int ith) {
    const int n = p->n_items, chunk = p->chunk;
    for (;;) {
        const int b = atomic_fetch_add_explicit(&p->next, chunk, memory_order_relaxed);
        if (b >= n) break;
        const int e = b + chunk < n ? b + chunk : n;
        p->fn(p->ctx, b, e, ith);
        atomic_fetch_add_explicit(&p->done, e - b, memory_order_release);
    }
}

static void *worker_main(void *arg) {
    worker_arg *wa = (worker_arg *)arg;
    borg_pool *p = wa->pool;
    uint32_t seen = 0;
    for (;;) {
        uint64_t s = atomic_load_explicit(&p->state, memory_order_acquire);
        long spins = 0;
        while (ST_GEN(s) == seen && !atomic_load_explicit(&p->stop, memory_order_relaxed)) {
            if (spins < SPIN_ITERS) {
                cpu_relax();
            } else if (spins < SPIN_ITERS + YIELD_ITERS) {
                sched_yield();
            } else {
                pthread_mutex_lock(&p->mu);
                atomic_fetch_add(&p->sleeping, 1);
                while (ST_GEN(atomic_load(&p->state)) == seen && !atomic_load(&p->stop))
                    pthread_cond_wait(&p->cv, &p->mu);
                atomic_fetch_sub(&p->sleeping, 1);
                pthread_mutex_unlock(&p->mu);
                spins = 0;
            }
            spins++;
            s = atomic_load_explicit(&p->state, memory_order_acquire);
        }
        if (atomic_load(&p->stop)) break;
        const uint32_t g = ST_GEN(s);
        seen = g;
        /* join unless the dispatch already closed */
        int joined = 0;
        while (ST_GEN(s) == g && !(s & ST_CLOSED)) {
            if (atomic_compare_exchange_weak(&p->state, &s, s + 1)) {
                joined = 1;
                break;
            }
        }
        if (!joined) continue;
        run_chunks(p, wa->ith);
        atomic_fetch_sub_explicit(&p->state, 1, memory_order_release);
    }
    return NULL;
}

static int default_threads(void) {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    long n = (long)si.dwNumberOfProcessors;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
#endif
    if (n < 1) n = 1;
    if (n > 64) n = 64;
    return (int)n;
}

borg_pool *borg_pool_new(int n_threads) {
    if (n_threads <= 0) n_threads = default_threads();
    borg_pool *p = (borg_pool *)calloc(1, sizeof(borg_pool));
    if (!p) return NULL;
    p->n = n_threads;
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->cv, NULL);
    if (n_threads > 1) {
        p->threads = (pthread_t *)calloc((size_t)n_threads, sizeof(pthread_t));
        p->args = (worker_arg *)calloc((size_t)n_threads, sizeof(worker_arg));
        if (!p->threads || !p->args) {
            free(p->threads);
            free(p->args);
            free(p);
            return NULL;
        }
        for (int i = 1; i < n_threads; i++) {
            p->args[i].pool = p;
            p->args[i].ith = i;
            if (pthread_create(&p->threads[i], NULL, worker_main, &p->args[i]) != 0) {
                p->n = i; /* run with the threads we got */
                break;
            }
        }
    }
    return p;
}

void borg_pool_free(borg_pool *p) {
    if (!p) return;
    pthread_mutex_lock(&p->mu);
    atomic_store(&p->stop, 1);
    pthread_cond_broadcast(&p->cv);
    pthread_mutex_unlock(&p->mu);
    for (int i = 1; i < p->n; i++) pthread_join(p->threads[i], NULL);
    pthread_mutex_destroy(&p->mu);
    pthread_cond_destroy(&p->cv);
    free(p->threads);
    free(p->args);
    free(p);
}

int borg_pool_size(const borg_pool *p) { return p ? p->n : 1; }

static void wait_for(atomic_int *v, int target) {
    long spins = 0;
    while (atomic_load_explicit(v, memory_order_acquire) < target) {
        if (++spins < SPIN_ITERS) cpu_relax();
        else sched_yield(); /* a preempted worker may need this core */
    }
}

void pool_for(borg_pool *p, int n_items, int chunk, borg_range_fn fn, void *ctx) {
    if (n_items <= 0) return;
    if (chunk < 1) chunk = 1;
    if (!p || p->n <= 1 || n_items <= chunk) {
        fn(ctx, 0, n_items, 0);
        return;
    }
    p->fn = fn;
    p->ctx = ctx;
    p->n_items = n_items;
    p->chunk = chunk;
    atomic_store_explicit(&p->next, 0, memory_order_relaxed);
    atomic_store_explicit(&p->done, 0, memory_order_relaxed);
    /* open the next generation; previous dispatch is closed with zero joiners */
    uint64_t s = atomic_load(&p->state);
    atomic_store(&p->state, ((uint64_t)(ST_GEN(s) + 1) << 32));
    if (atomic_load(&p->sleeping) > 0) {
        pthread_mutex_lock(&p->mu);
        pthread_cond_broadcast(&p->cv);
        pthread_mutex_unlock(&p->mu);
    }
    run_chunks(p, 0);
    wait_for(&p->done, n_items);
    /* close, then wait for joined workers to leave so p->ctx can go out of scope */
    atomic_fetch_or(&p->state, ST_CLOSED);
    long spins = 0;
    while (ST_ACTIVE(atomic_load_explicit(&p->state, memory_order_acquire)) != 0) {
        if (++spins < SPIN_ITERS) cpu_relax();
        else sched_yield();
    }
}

#endif
