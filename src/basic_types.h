#ifndef BASIC_TYPES_H
#define BASIC_TYPES_H

#include <cstddef>
#include <iterator>

// array_t

template<typename ElementType>
class array_t {
public:
    size_t const size;

private:
    ElementType *const memory;

public:
    array_t(ElementType *const memory, size_t const size);

    ElementType *begin();
    ElementType const *begin() const;
    ElementType *end();
    ElementType const *end() const;
};

template<typename ElementType>
ElementType *array_t<ElementType>::begin() {
    return memory;
}

template<typename ElementType>
ElementType *array_t<ElementType>::end() {
    return memory+size;
}

template<typename ElementType>
ElementType const *array_t<ElementType>::begin() const {
    return memory;
}

template<typename ElementType>
ElementType const *array_t<ElementType>::end() const {
    return memory+size;
}

template<typename ElementType>
array_t<ElementType>::array_t(ElementType *const memory, size_t const size) : size(size), memory(memory) {
}

// matrix

template<typename ElementType>
class matrix {
public:
    class iterator;
    size_t const height, width;

private:
    ElementType *const memory = new ElementType[height * width];

public:
    matrix(size_t const height, size_t const width);
    ~matrix();
    ElementType *operator[](size_t const i);
    ElementType const *data();

    iterator begin();
    iterator end();
    iterator const begin() const;
    iterator const end() const;
};

template<typename ElementType>
typename matrix<ElementType>::iterator matrix<ElementType>::begin() {
    return iterator(height, width, 0, memory);
}

template<typename ElementType>
typename matrix<ElementType>::iterator matrix<ElementType>::end() {
    return iterator(height, width, height, memory);
}

template<typename ElementType>
typename matrix<ElementType>::iterator const matrix<ElementType>::begin() const {
    return iterator(height, width, 0, memory);
}

template<typename ElementType>
typename matrix<ElementType>::iterator const matrix<ElementType>::end() const {
    return iterator(height, width, height, memory);
}

template<typename ElementType>
matrix<ElementType>::matrix(size_t const height, size_t const width) : height(height), width(width) {
}

template<typename ElementType>
matrix<ElementType>::~matrix() {
    delete[] memory;
}

template<typename ElementType>
ElementType *matrix<ElementType>::operator[](size_t const i) {
    return &memory[width*i];
}

template<typename ElementType>
ElementType const *matrix<ElementType>::data() {
    return memory;
}


// Iterator

template<typename ElementType>
class matrix<ElementType>::iterator : public std::iterator<
    std::random_access_iterator_tag,        // iterator_category
    array_t<ElementType>,                   // value_type
    int64_t,                                // difference_type
    void *,                                 // pointer
    array_t<ElementType>                    // reference
> {
    friend matrix<ElementType>;
    size_t const height, width;
    size_t position;
    ElementType *const memory;

    public:
    explicit iterator(size_t const height, size_t const width,
        size_t const position, ElementType *const memory) : height(height),
        width(width), position(position), memory(memory) {
    }

    bool operator==(iterator const &b) {
        return b.position == position;
    }

    bool operator!=(iterator const &b) {
        return !(*this == b);
    }

    array_t<ElementType> operator*() {
        return array_t<ElementType>(memory + width*position, width);
    }

    iterator& operator++() {
        position += 1;

        return *this;
    }
};

#endif
