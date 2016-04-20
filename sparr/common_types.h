#ifndef SP_MAP_TYPES
#define SP_MAP_TYPES

namespace sparray {

typedef long int single_index_type;  // FIXME: npy_intp, likely Py_ssize_t


/* 
 * Store the result of:
 *      PySlice_IndicesEx,
 *      reduce start and stop to += length, if necessary
 */
struct slice_t
{
    single_index_type start, stop, step;
    single_index_type slicelength;
};

} // namespace sparray

#endif
