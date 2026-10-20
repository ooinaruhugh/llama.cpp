#include "llama-backing-store.h"

#include "llama-impl.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>

// one implementation per platform, a build can force one (or LLAMA_BACKING_STORE_NONE) for compile checks
#if !defined(LLAMA_BACKING_STORE_SHM) && !defined(LLAMA_BACKING_STORE_MEMFD) && !defined(LLAMA_BACKING_STORE_NONE)
#    if defined(__APPLE__)
#        define LLAMA_BACKING_STORE_SHM
#    elif defined(__linux__)
#        define LLAMA_BACKING_STORE_MEMFD
#    endif
#endif

#if defined(LLAMA_BACKING_STORE_SHM) || defined(LLAMA_BACKING_STORE_MEMFD)

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef LLAMA_BACKING_STORE_MEMFD
#include <sys/syscall.h>

#ifndef MFD_CLOEXEC
#define MFD_CLOEXEC       0x0001U
#endif
#ifndef MFD_ALLOW_SEALING
#define MFD_ALLOW_SEALING 0x0002U
#endif
#ifndef F_ADD_SEALS
#define F_ADD_SEALS       (1024 + 9)
#endif
#ifndef F_SEAL_SEAL
#define F_SEAL_SEAL       0x0001
#define F_SEAL_SHRINK     0x0002
#define F_SEAL_GROW       0x0004
#define F_SEAL_WRITE      0x0008
#endif
#endif

struct llama_backing_store::impl {
    size_t size   = 0;
    int    fd_w   = -1;      // writer descriptor, closed by seal()
    int    fd_r   = -1;      // descriptor handed out after seal()
    void * addr_w = nullptr; // writer mapping
    bool   sealed = false;
    bool   ro_enforced = false;

    static void set_cloexec(int fd) {
        const int flags = fcntl(fd, F_GETFD);
        if (flags == -1 || fcntl(fd, F_SETFD, flags | FD_CLOEXEC) == -1) {
            throw std::runtime_error(format("cannot set close-on-exec: %s", strerror(errno)));
        }
    }

    void map_writer() {
        void * p = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd_w, 0);
        if (p == MAP_FAILED) {
            throw std::runtime_error(format("cannot map %zu bytes of anonymous memory: %s", size, strerror(errno)));
        }
        addr_w = p;
    }

    void unmap_writer() {
        if (addr_w) {
            munmap(addr_w, size);
            addr_w = nullptr;
        }
    }

    void release() {
        unmap_writer();
        if (fd_w != -1) {
            close(fd_w);
        }
        if (fd_r != -1 && fd_r != fd_w) {
            close(fd_r);
        }
        fd_w = fd_r = -1;
    }

#if defined(LLAMA_BACKING_STORE_SHM)
    // POSIX shm: the name is unlinked at once, so the object lives only as long as descriptors and mappings
    // ftruncate works once, the second descriptor is opened O_RDONLY so that seal() can drop the writable one
    explicit impl(size_t size) : size(size) {
        char name[32]; // macOS limit is 31 characters
        for (int attempt = 0; ; attempt++) {
            snprintf(name, sizeof(name), "/llama.%d.%08x", (int) getpid(), (unsigned) arc4random());
            fd_w = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
            if (fd_w != -1) {
                break;
            }
            if (errno != EEXIST || attempt >= 8) {
                throw std::runtime_error(format("shm_open failed: %s", strerror(errno)));
            }
        }

        try {
            set_cloexec(fd_w);
            if (ftruncate(fd_w, (off_t) size) == -1) {
                throw std::runtime_error(format("cannot allocate %zu bytes of shared memory: %s", size, strerror(errno)));
            }
            map_writer();
            fd_r = shm_open(name, O_RDONLY, 0);
            if (fd_r == -1) {
                throw std::runtime_error(format("shm_open (read-only) failed: %s", strerror(errno)));
            }
            set_cloexec(fd_r);
        } catch (...) {
            shm_unlink(name);
            release();
            throw;
        }
        shm_unlink(name);
    }

    void seal() {
        unmap_writer();
        if (fd_w != -1) {
            close(fd_w);
            fd_w = -1;
        }
        ro_enforced = true;
    }
#else
    // memfd: sealing makes the contents immutable. F_SEAL_WRITE fails while a writable mapping exists, so unmap first
    explicit impl(size_t size) : size(size) {
#ifndef SYS_memfd_create
        throw std::runtime_error("memfd_create is not available in this build");
#else
        fd_w = (int) syscall(SYS_memfd_create, "llama-model", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        if (fd_w == -1) {
            throw std::runtime_error(format("memfd_create failed: %s", strerror(errno)));
        }
        try {
            if (ftruncate(fd_w, (off_t) size) == -1) {
                throw std::runtime_error(format("cannot allocate %zu bytes of memory: %s", size, strerror(errno)));
            }
            // reserve the pages now, so exhaustion is an error here and not a fault while decoding
            if (fallocate(fd_w, 0, 0, (off_t) size) == -1 && errno != EOPNOTSUPP && errno != ENOSYS) {
                throw std::runtime_error(format("cannot reserve %zu bytes of memory: %s", size, strerror(errno)));
            }
            map_writer();
        } catch (...) {
            release();
            throw;
        }
        fd_r = fd_w;
#endif
    }

    void seal() {
        unmap_writer();
        if (fcntl(fd_r, F_ADD_SEALS, F_SEAL_SHRINK | F_SEAL_GROW | F_SEAL_WRITE | F_SEAL_SEAL) == -1) {
            LLAMA_LOG_WARN("%s: cannot seal the memory object: %s\n", __func__, strerror(errno));
        } else {
            ro_enforced = true;
        }
        fd_w = -1; // same descriptor as fd_r
    }
#endif

    ~impl() {
        release();
    }
};

std::unique_ptr<llama_backing_store> llama_backing_store::create(size_t size) {
    if (size == 0) {
        throw std::runtime_error("cannot create an empty backing store");
    }
    if (size > (size_t) std::numeric_limits<off_t>::max()) {
        throw std::runtime_error(format("backing store size %zu does not fit in a file offset", size));
    }
    std::unique_ptr<llama_backing_store> bs(new llama_backing_store());
    bs->pimpl.reset(new impl(size));
    return bs;
}

void * llama_backing_store::write_addr() {
    if (pimpl->sealed) {
        throw std::runtime_error("backing store is sealed");
    }
    return pimpl->addr_w;
}

void llama_backing_store::seal() {
    if (!pimpl->sealed) {
        pimpl->seal();
        pimpl->sealed = true;
    }
}

bool llama_backing_store::read_only_enforced() const {
    return pimpl->ro_enforced;
}

int llama_backing_store::fd() const {
    if (!pimpl->sealed) {
        throw std::runtime_error("backing store is not sealed");
    }
    return pimpl->fd_r;
}

#else // no implementation for this platform

struct llama_backing_store::impl {
    size_t size = 0;
};

std::unique_ptr<llama_backing_store> llama_backing_store::create(size_t) {
    throw std::runtime_error("compressed models are not supported on this platform (no anonymous memory backing store)");
}

void * llama_backing_store::write_addr()              { return nullptr; }
void   llama_backing_store::seal()                    {}
bool   llama_backing_store::read_only_enforced() const { return false; }
int    llama_backing_store::fd() const                { return -1; }

#endif

size_t llama_backing_store::size() const {
    return pimpl->size;
}

llama_backing_store::~llama_backing_store() = default;
