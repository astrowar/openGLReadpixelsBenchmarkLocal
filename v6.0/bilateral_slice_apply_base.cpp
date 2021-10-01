#include "bilateral_slice_apply.h"

#include <GL/glew.h>
#include <GL/freeglut.h>



#include <string>
#include <iostream>
#include <vector>
#include "bilateral_slice_data.h"

namespace hdrnet
{

    namespace bilateral_slice_apply
    {

        bilateral_slice_apply_data *create_bilateral_slice_apply_data(
            int output_width,
            int output_height,
            int grid_width,
            int grid_height,
            int grid_depth,
            int input_chans,
            int output_chans,
            bool has_offset)
        {
            bilateral_slice_apply_data *data = new bilateral_slice_apply_data();
            data->output_width_ = output_width;
            data->output_height_ = output_height;
            data->input_width_ = output_width;
            data->input_height_ = output_height;

            data->grid_width_ = grid_width;
            data->grid_height_ = grid_height;
            data->grid_depth_ = grid_depth;

            data->input_chans = input_chans;
            data->output_chans = output_chans;
            data->has_offset = has_offset;

            return data;
        }

        void replaceAll(std::string &str, const std::string &from, const std::string &to)
        {
            if (from.empty())
                return;
            size_t start_pos = 0;
            while ((start_pos = str.find(from, start_pos)) != std::string::npos)
            {
                str.replace(start_pos, from.length(), to);
                start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
            }
        }

        void shader_from_file(std::string &content, GLuint &shader)
        {

            const GLchar *source = (const GLchar *)content.c_str();
            glShaderSource(shader, 1, &source, NULL);
            glCompileShader(shader);

            GLint shader_success;
            glGetShaderiv(shader, GL_COMPILE_STATUS, &shader_success);
            if (shader_success == GL_FALSE)
            {
                std::cout << "Failed to compile shader" << std::endl;
                GLint logSize = 0;
                glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &logSize);
                std::vector<GLchar> errorLog(logSize);
                glGetShaderInfoLog(shader, logSize, &logSize, &errorLog[0]);
                std::cout << errorLog.data() << std::endl;
                glDeleteShader(shader);
                throw;
            }
        }

        int init(bilateral_slice_apply_data *data)
        {

            std::string vertex_shader_content = R"(  #version 400

in vec2 vPosition;
in vec2 vTexCoord;
out vec2 texCoord;

void main() {
  vec4 homogeneous = vec4(vTexCoord.xy, 0, 1);
  texCoord = vTexCoord;
  gl_Position = vec4(vPosition.x, vPosition.y, 0.0, 1.0);
}
 )";
            std::string fragment_shader_content = R"(


#version 430
precision highp float;

layout(location = 0) out vec3 colorOut;
 layout(location = 1) uniform sampler2D input_texture;
 layout(location = 2) uniform sampler2D guide_texture;
 layout(location = 3) uniform sampler2D grid_texture;

in vec2 texCoord;

const int bs = 1;
const int h = $h ;
const int w = $w ;
const int gh = $gh ;
const int gw = $gw  ;
const int gd =  $gd ;
const int input_chans = $input_chans;
const int output_chans = $output_chans;
const bool has_offset = true;

float diff_abs(float x) {
  float eps = 1e-8;
  return sqrt(x * x + eps);
}

float d_diff_abs(float x) {
  float eps = 1e-8;
  return x / sqrt(x * x + eps);
}

float weight_z(float x) {
  float abx = diff_abs(x);
  return max(1.0 - abx, 0.0);
}

void main() {

  int grid_chans = input_chans * output_chans;
  int coeff_stride = input_chans;
  if(has_offset) {
    grid_chans += output_chans;
    coeff_stride += 1;
  }

  // X and Y are the imagem cordinades of actual fragment in PIXELS
  int x = int(gl_FragCoord.x);
  int y = int(gl_FragCoord.y);

  int b = 0;  // batch actual is 1

  float gx = (x + 0.5f) * gw / (1.0f * w);
  float gy = (y + 0.5f) * gh / (1.0f * h);

  float gz = texture(guide_texture,texCoord).x * gd;  //red channle only for 1 channel textures

  int fx = int(floor(gx - 0.5f));
  int fy = int(floor(gy - 0.5f));
  int fz = int(floor(gz - 0.5f));

    // Grid strides
  int sz = grid_chans;
  int sx = grid_chans * gd;
  int sy = grid_chans * gd * gw;
  int sb = grid_chans * gd * gw * gh;

 vec3 colorOut_tmp = vec3(0.0f, 0.0f, 0.0f);
  for(int out_c = 0; out_c < output_chans; out_c =out_c +1 )
  {

    float value = 0.0f;

    for(int in_c = 0; in_c < coeff_stride; in_c= in_c+1) {
      float coeff_sample = 0.0f;

      for(int xx = fx; xx < fx + 2; xx += 1 ) {
        int x_ = max(min(xx, gw - 1), 0);
        float wx = max(1.0f - abs(xx + 0.5f - gx), 0.0);
        for(int yy = fy; yy < fy + 2; yy +=1) {
          int y_ = max(min(yy, gh - 1), 0);
          float wy = max(1.0f - abs(yy + 0.5f - gy), 0.0);
          for(int zz = fz; zz < fz + 2; zz+=1 ) {
            int z_ = max(min(zz, gd - 1), 0);
            float wz = weight_z(zz + 0.5f - gz);

              int tx =  (coeff_stride * out_c + in_c) + (sz * z_) + (sx * x_);
              vec2 gxyz =  vec2( float(tx+0.5)/float(sy) ,  float(y_ + 0.5 )/float(gh)   );
              coeff_sample +=  (texture(grid_texture, gxyz ).x)* wx * wy * wz;
          }
        }
      }

    // Grid trilinear interpolation
      if(in_c < input_chans) {
        vec3 value_t = coeff_sample* texture(input_texture, texCoord).xyz;
        value += value_t[in_c]  ;
      } else { // Offset term
        value += coeff_sample;
      }
    }  //in_c loop
      colorOut_tmp[out_c] = value   ;
  } // out_c loop
  colorOut =   colorOut_tmp;
}

             )";

            replaceAll(fragment_shader_content, "$w", std::to_string(data->input_width_));
            replaceAll(fragment_shader_content, "$h", std::to_string(data->input_height_));
            replaceAll(fragment_shader_content, "$gw", std::to_string(data->grid_width_));
            replaceAll(fragment_shader_content, "$gh", std::to_string(data->grid_height_));
            replaceAll(fragment_shader_content, "$gd", std::to_string(data->grid_depth_));
            replaceAll(fragment_shader_content, "$input_chans", std::to_string(data->input_chans));
            replaceAll(fragment_shader_content, "$output_chans", std::to_string(data->output_chans));

            int g_chans  = data->input_chans * data->output_chans;
            if (data->has_offset) g_chans  = (1 + data->input_chans) * data->output_chans;
            replaceAll(fragment_shader_content, "$g_chans", std::to_string( g_chans));

            std::cout << fragment_shader_content << std::endl;

            data->vertex_shader_ = glCreateShader(GL_VERTEX_SHADER);
            data->fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
            shader_from_file(vertex_shader_content, data->vertex_shader_);
            shader_from_file(fragment_shader_content, data->fragment_shader_);

            // Create program
            data->program_ = glCreateProgram();
            glAttachShader(data->program_, data->vertex_shader_);
            glAttachShader(data->program_, data->fragment_shader_);
            glLinkProgram(data->program_);
            GLint link_success;
            glGetProgramiv(data->program_, GL_LINK_STATUS, &link_success);
            if (link_success == GL_FALSE)
            {
                std::cout << "Failed to link program" << std::endl;
                GLint logSize = 0;
                glGetProgramiv(data->program_, GL_INFO_LOG_LENGTH, &logSize);
                std::vector<GLchar> errorLog(logSize);
                glGetProgramInfoLog(data->program_, logSize, &logSize, &errorLog[0]);
                std::cout << errorLog.data() << std::endl;
                glDeleteProgram(data->program_);
                data->program_ = 0;
                throw;
            }
            glUseProgram(data->program_);

            // Geometry
            int ph = glGetAttribLocation(data->program_, "vPosition");
            int tch = glGetAttribLocation(data->program_, "vTexCoord");
            glVertexAttribPointer(ph, 2, GL_FLOAT, false, 4 * 2, static_cast<GLvoid *>(data->pVertex_));
            glVertexAttribPointer(tch, 2, GL_FLOAT, false, 4 * 2, static_cast<GLvoid *>(data->pTexCoord_));
            glEnableVertexAttribArray(ph);
            glEnableVertexAttribArray(tch);

            // Output texture
            glGenTextures(1, &data->output_texture_);
            glBindTexture(GL_TEXTURE_2D, data->output_texture_);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, data->output_width_, data->output_height_, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

            // Input texture
            glGenTextures(1, &data->input_texture_);
            glBindTexture(GL_TEXTURE_2D, data->input_texture_);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, data->output_width_, data->output_height_);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);



            // guide texture
            glGenTextures(1, &data->guide_texture_);
            glBindTexture(GL_TEXTURE_2D, data->guide_texture_);
            //glTexStorage2D(GL_TEXTURE_2D, 1, GL_FLOAT , data->output_width_, data->output_height_);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, data->output_width_, data->output_height_);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);


            //grid Texture

           glGenTextures(1, &data->grid_texture_);
           glBindTexture(GL_TEXTURE_2D, data->grid_texture_);
           glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, data->grid_depth_ *  g_chans *data->grid_width_  ,data->grid_height_  );
           glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
           glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
           glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
           glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

            glUseProgram(data->program_);


            glProgramUniform1i(data->program_, glGetUniformLocation(data->program_, "input_texture"), 1);
            glProgramUniform1i(data->program_, glGetUniformLocation(data->program_, "guide_texture"), 2);
            glProgramUniform1i(data->program_, glGetUniformLocation(data->program_, "grid_texture"), 3);


            // Output framebuffer
            glGenFramebuffers(1, &data->framebuffer_);
            glBindFramebuffer(GL_FRAMEBUFFER, data->framebuffer_);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, data->output_texture_, 0);
            GLenum draw_buffers[1] = {GL_COLOR_ATTACHMENT0};
            glDrawBuffers(1, draw_buffers); // "1" is the size of DrawBuffers

            // Always check that our framebuffer is ok
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            {
                std::cout << "Frame buffer did not complete operation" << std::endl;
                throw;
            }

            glBindFramebuffer(GL_FRAMEBUFFER, data->framebuffer_);
            glViewport(0, 0, data->output_width_, data->output_height_);
            return 0; //no error
        }

        int eval(bilateral_slice_apply_data *data, const float *grid, const float *guide, const float *input, const float *output)
        {
            //int bs, int gh, int gw, int gd, int input_chans, int output_chans, bool has_offset, int h, int w, const float* const grid, const float* const guide, const float* const input,    float* const out

            //grid channels = (1+i_chans)*o_chans
            int grid_channels = data->input_chans * data->output_chans;
            if (data->has_offset)
            {
                grid_channels += data->output_chans;
            }

            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, data->input_texture_);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data->output_width_, data->output_height_, GL_RGB, GL_FLOAT, input);

            glActiveTexture(GL_TEXTURE2);
            glBindTexture(GL_TEXTURE_2D, data->guide_texture_);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data->output_width_, data->output_height_, GL_RED, GL_FLOAT, guide);

            glActiveTexture(GL_TEXTURE3);
            glBindTexture(GL_TEXTURE_2D, data->grid_texture_);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data->grid_depth_ * grid_channels * data->grid_width_ ,data->grid_height_   ,   GL_RED, GL_FLOAT, grid);


            glClear(GL_COLOR_BUFFER_BIT);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

            glReadPixels(0, 0, data->output_width_, data->output_height_, GL_RGB, GL_FLOAT, (GLvoid *)output);
            return 0;
        }

        int prepare(bilateral_slice_apply_data *data)
        {
            if (data->input_width_ != data->output_width_)
                return -1;
            if (data->input_height_ != data->output_height_)
                return -1;
            return 0;
        }

        int free(bilateral_slice_apply_data *data)
        {
            glDeleteTextures(1, &data->input_texture_);
            glDeleteTextures(1, &data->output_texture_);
            glDeleteTextures(1, &data->guide_texture_);
            glDeleteTextures(1, &data->grid_texture_);

            glDeleteShader(data->vertex_shader_);
            glDeleteShader(data->fragment_shader_);
            glDeleteProgram(data->program_);

            return 0;
        }
    }
}
