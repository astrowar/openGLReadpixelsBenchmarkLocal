

namespace hdrnet{

  namespace bilateral_slice_apply{

     class bilateral_slice_apply_data  ;

      bilateral_slice_apply_data*  create_bilateral_slice_apply_data(
            int output_width,
            int output_height,
            int grid_width,
            int grid_heigh,
            int grid_depth,
            int input_chans,
            int output_chans,
            bool has_offset
      );

      int init(bilateral_slice_apply_data* data);  //init all contexts, shaders , buffers and texture IDs
      int prepare(bilateral_slice_apply_data* data);  //fill buffers
      int eval(bilateral_slice_apply_data *data,  const float*   grid ,  const float*   guide , const float*  input ,  const float*  output );          //execute shader
      int free(bilateral_slice_apply_data* data);  // free all contexts, shaders, buffers ans textureIds

  }

}