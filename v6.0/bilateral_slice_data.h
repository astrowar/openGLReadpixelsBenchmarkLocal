#include <GL/gl.h>

namespace hdrnet{

  namespace bilateral_slice_apply{

      class bilateral_slice_apply_data
        {

        public:
            int input_width_;
            int input_height_;

            int output_width_;
            int output_height_;

            int grid_width_;
            int grid_height_;
            int grid_depth_;

            int input_chans;
            int output_chans;
            bool has_offset;



            //guide width == input width
            //guide height == input height

            GLuint program_;
            GLuint vertex_shader_;
            GLuint fragment_shader_;

            GLuint guide_texture_;
            GLuint grid_texture_ ;
            GLuint input_texture_;

            GLuint output_texture_;
            GLuint framebuffer_;

            GLuint ssbo;
            float ssbo_buffer[25000];

            GLuint m_ids[2];
            GLuint m_active;

            float pVertex_[8] = {1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f};
            float pTexCoord_[8] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f};
        };

  }
}