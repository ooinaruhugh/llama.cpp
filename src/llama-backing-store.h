#pragma once

#include <cstddef>
#include <memory>

// Anonymous, memory-backed object that holds an uncompressed model.
//
//   create(size) -> a decoder fills write_addr() -> seal() -> llama_mmap maps fd() read-only
//
// Only fstat and mmap on fd() are guaranteed, read/seek may fail (POSIX shm on macOS).
// The pages stay alive while any mapping exists, also after the store itself is destroyed.
// The store knows nothing about codecs or GGUF.
struct llama_backing_store {
    // throws std::runtime_error if the platform has no implementation or the memory cannot be set up
    static std::unique_ptr<llama_backing_store> create(size_t size);

    ~llama_backing_store();

    size_t size() const;

    // single writable view of the whole object, valid until seal()
    void * write_addr();

    // drops the writer view and the write capability, contents are final afterwards
    void seal();

    // true if the OS refuses writable mappings of fd() after seal()
    bool read_only_enforced() const;

    // close-on-exec descriptor, valid after seal()
    int fd() const;

private:
    llama_backing_store() = default;

    struct impl;
    std::unique_ptr<impl> pimpl;
};
