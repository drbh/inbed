//! Lock-free primitives for the request path: a bounded multi-producer ring (Vyukov's per-slot
//! sequence-number queue) and a one-shot reply slot woken by `park`/`unpark`. No mutexes.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::Thread;

/// Pads a value to its own cache line so producer and consumer counters don't false-share.
#[repr(align(128))]
struct Padded<T>(T);

struct Slot<T> {
    seq: AtomicUsize,
    val: UnsafeCell<MaybeUninit<T>>,
}

/// Bounded ring. Any number of threads may `push`; `pop` is also safe from several threads, but
/// inbed only pops from the model's worker.
pub struct Ring<T> {
    buf: Box<[Slot<T>]>,
    mask: usize,
    head: Padded<AtomicUsize>,
    tail: Padded<AtomicUsize>,
}

unsafe impl<T: Send> Send for Ring<T> {}
unsafe impl<T: Send> Sync for Ring<T> {}

impl<T> Ring<T> {
    /// Capacity rounds up to a power of two (at least 2).
    pub fn new(cap: usize) -> Ring<T> {
        let cap = cap.max(2).next_power_of_two();
        let buf = (0..cap)
            .map(|i| Slot {
                seq: AtomicUsize::new(i),
                val: UnsafeCell::new(MaybeUninit::uninit()),
            })
            .collect();
        Ring {
            buf,
            mask: cap - 1,
            head: Padded(AtomicUsize::new(0)),
            tail: Padded(AtomicUsize::new(0)),
        }
    }

    pub fn capacity(&self) -> usize {
        self.mask + 1
    }

    /// `Err(v)` when full.
    pub fn push(&self, v: T) -> Result<(), T> {
        let mut pos = self.tail.0.load(Ordering::Relaxed);
        loop {
            let slot = &self.buf[pos & self.mask];
            let seq = slot.seq.load(Ordering::Acquire);
            let dif = seq as isize - pos as isize;
            if dif == 0 {
                match self.tail.0.compare_exchange_weak(
                    pos,
                    pos + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        unsafe { (*slot.val.get()).write(v) };
                        slot.seq.store(pos + 1, Ordering::Release);
                        return Ok(());
                    }
                    Err(p) => pos = p,
                }
            } else if dif < 0 {
                return Err(v);
            } else {
                pos = self.tail.0.load(Ordering::Relaxed);
            }
        }
    }

    pub fn pop(&self) -> Option<T> {
        let mut pos = self.head.0.load(Ordering::Relaxed);
        loop {
            let slot = &self.buf[pos & self.mask];
            let seq = slot.seq.load(Ordering::Acquire);
            let dif = seq as isize - (pos + 1) as isize;
            if dif == 0 {
                match self.head.0.compare_exchange_weak(
                    pos,
                    pos + 1,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => {
                        let v = unsafe { (*slot.val.get()).assume_init_read() };
                        slot.seq.store(pos + self.mask + 1, Ordering::Release);
                        return Some(v);
                    }
                    Err(p) => pos = p,
                }
            } else if dif < 0 {
                return None;
            } else {
                pos = self.head.0.load(Ordering::Relaxed);
            }
        }
    }

    /// A snapshot: may be stale by the time the caller acts on it.
    pub fn is_empty(&self) -> bool {
        let pos = self.head.0.load(Ordering::Acquire);
        let seq = self.buf[pos & self.mask].seq.load(Ordering::Acquire);
        seq != pos + 1
    }
}

impl<T> Drop for Ring<T> {
    fn drop(&mut self) {
        while self.pop().is_some() {}
    }
}

struct Cell<R> {
    done: AtomicBool,
    val: UnsafeCell<Option<R>>,
    waiter: Thread,
}

unsafe impl<R: Send> Send for Cell<R> {}
unsafe impl<R: Send> Sync for Cell<R> {}

/// The receiving half of a one-shot reply. Only the thread that created it may `wait`.
pub struct Ticket<R>(Arc<Cell<R>>);

/// The sending half. Dropping it without `send` delivers `on_drop()` so a waiter never hangs.
pub struct Promise<R> {
    cell: Option<Arc<Cell<R>>>,
    on_drop: fn() -> R,
}

/// A reply slot whose waiter is the current thread.
pub fn oneshot<R>(on_drop: fn() -> R) -> (Promise<R>, Ticket<R>) {
    let c = Arc::new(Cell {
        done: AtomicBool::new(false),
        val: UnsafeCell::new(None),
        waiter: std::thread::current(),
    });
    (
        Promise {
            cell: Some(c.clone()),
            on_drop,
        },
        Ticket(c),
    )
}

impl<R> Promise<R> {
    pub fn send(mut self, r: R) {
        if let Some(c) = self.cell.take() {
            deliver(c, r);
        }
    }
}

fn deliver<R>(c: Arc<Cell<R>>, r: R) {
    // the waiter reads `val` only after observing `done` (Acquire), so this write is exclusive
    unsafe { *c.val.get() = Some(r) };
    c.done.store(true, Ordering::Release);
    c.waiter.unpark();
}

impl<R> Drop for Promise<R> {
    fn drop(&mut self) {
        if let Some(c) = self.cell.take() {
            deliver(c, (self.on_drop)());
        }
    }
}

impl<R> Ticket<R> {
    pub fn wait(self) -> R {
        debug_assert_eq!(self.0.waiter.id(), std::thread::current().id());
        // park can return spuriously, and unpark may arrive before park: both are covered by
        // re-checking `done`, and the park token is sticky
        while !self.0.done.load(Ordering::Acquire) {
            std::thread::park();
        }
        unsafe { (*self.0.val.get()).take().expect("reply delivered once") }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_basic() {
        let r = Ring::new(3);
        assert_eq!(r.capacity(), 4);
        assert!(r.is_empty());
        for i in 0..4 {
            r.push(i).unwrap();
        }
        assert_eq!(r.push(9), Err(9));
        assert!(!r.is_empty());
        for i in 0..4 {
            assert_eq!(r.pop(), Some(i));
        }
        assert_eq!(r.pop(), None);
        assert!(r.is_empty());
    }

    #[test]
    fn ring_drops_contents() {
        let a = Arc::new(());
        {
            let r = Ring::new(8);
            for _ in 0..5 {
                r.push(a.clone()).unwrap();
            }
            r.pop();
        }
        assert_eq!(Arc::strong_count(&a), 1);
    }

    /// Many producers, one consumer: every value arrives exactly once, and each producer's values
    /// arrive in its own order.
    #[test]
    fn ring_stress_mpsc() {
        const P: usize = 8;
        const N: usize = 200_000;
        let r = Arc::new(Ring::<(usize, usize)>::new(64));
        let hs: Vec<_> = (0..P)
            .map(|p| {
                let r = r.clone();
                std::thread::spawn(move || {
                    for i in 0..N {
                        let mut v = (p, i);
                        while let Err(back) = r.push(v) {
                            v = back;
                            std::hint::spin_loop();
                        }
                    }
                })
            })
            .collect();
        let mut next = [0usize; P];
        let mut got = 0;
        while got < P * N {
            if let Some((p, i)) = r.pop() {
                assert_eq!(i, next[p], "producer {p} out of order");
                next[p] += 1;
                got += 1;
            } else {
                std::hint::spin_loop();
            }
        }
        for h in hs {
            h.join().unwrap();
        }
        assert!(r.pop().is_none());
    }

    #[test]
    fn oneshot_across_threads() {
        for i in 0..2000 {
            let (p, t) = oneshot::<usize>(|| usize::MAX);
            let h = std::thread::spawn(move || p.send(i));
            assert_eq!(t.wait(), i);
            h.join().unwrap();
        }
        let (p, t) = oneshot::<usize>(|| usize::MAX);
        std::thread::spawn(move || drop(p)).join().unwrap();
        assert_eq!(t.wait(), usize::MAX);
    }
}
