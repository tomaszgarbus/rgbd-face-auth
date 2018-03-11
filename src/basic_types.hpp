#ifndef BASIC_TYPES_HPP
#define BASIC_TYPES_HPP

#include <cstddef>
#include <iterator>

// Declarations

template <typename ElementType>
class Array {
 public:
   Array(ElementType *memory, size_t size);

   ElementType *begin();
   ElementType *end();
   ElementType const *begin() const;
   ElementType const *end() const;

   size_t const size;

 private:
   ElementType *const memory;
};

template <typename ElementType>
class Matrix {
 public:
   Matrix(size_t height, size_t width);
   Matrix(const Matrix &src);
   ~Matrix();

   ElementType *operator[](size_t i);
   ElementType *data();
   ElementType const *data() const;

   class iterator;

   Matrix<ElementType>::iterator begin();
   Matrix<ElementType>::iterator end();
   Matrix<ElementType>::iterator const begin() const;
   Matrix<ElementType>::iterator const end() const;

   size_t const height, width;

 private:
   ElementType *const memory = new ElementType[height * width];
};

// Definitions - Array

template <typename ElementType>
Array<ElementType>::Array(ElementType *const memory, size_t const size) : size(size), memory(memory) {}

template <typename ElementType>
ElementType *Array<ElementType>::begin() {
   return memory;
}

template <typename ElementType>
ElementType *Array<ElementType>::end() {
   return memory + size;
}

template <typename ElementType>
ElementType const *Array<ElementType>::begin() const {
   return memory;
}

template <typename ElementType>
ElementType const *Array<ElementType>::end() const {
   return memory + size;
}

// Definitions - Matrix

template <typename ElementType>
Matrix<ElementType>::Matrix(size_t const height, size_t const width) : height(height), width(width) {}

template <typename ElementType>
Matrix<ElementType>::Matrix(const Matrix &src) : height(src.height), width(src.width) {
   memcpy(memory, src.memory, height * width * sizeof(ElementType));
}

template <typename ElementType>
Matrix<ElementType>::~Matrix() {
   delete[] memory;
}

template <typename ElementType>
ElementType *Matrix<ElementType>::operator[](size_t const i) {
   return &memory[width * i];
}

template <typename ElementType>
ElementType *Matrix<ElementType>::data() {
   return memory;
}

template <typename ElementType>
ElementType const *Matrix<ElementType>::data() const {
   return memory;
}

template <typename ElementType>
typename Matrix<ElementType>::iterator Matrix<ElementType>::begin() {
   return iterator(height, width, 0, memory);
}

template <typename ElementType>
typename Matrix<ElementType>::iterator Matrix<ElementType>::end() {
   return iterator(height, width, height, memory);
}

template <typename ElementType>
typename Matrix<ElementType>::iterator const Matrix<ElementType>::begin() const {
   return iterator(height, width, 0, memory);
}

template <typename ElementType>
typename Matrix<ElementType>::iterator const Matrix<ElementType>::end() const {
   return iterator(height, width, height, memory);
}

// Matrix::iterator

template <typename ElementType>
class Matrix<ElementType>::iterator
      : public std::iterator<std::random_access_iterator_tag, Array<ElementType>, int64_t, void *, Array<ElementType>> {
   //                    <iterator_category,               value_type,
   //                    difference_type, pointer, reference>
   friend Matrix<ElementType>;
   size_t const height, width;
   size_t position;
   ElementType *const memory;

 public:
   explicit iterator(size_t const height, size_t const width, size_t const position, ElementType *const memory)
         : height(height), width(width), position(position), memory(memory) {}

   bool operator==(iterator const &b) {
      return b.position == position;
   }

   bool operator!=(iterator const &b) {
      return !(*this == b);
   }

   Array<ElementType> operator*() {
      return Array<ElementType>(memory + width * position, width);
   }

   iterator &operator++() {
      position += 1;
      return *this;
   }
};

#endif
