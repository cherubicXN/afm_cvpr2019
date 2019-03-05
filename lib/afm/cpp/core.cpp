#include <vector>
#include <iostream>
#include <stdio.h>
#include "afm.hpp"
#include <math.h>

float sgn(float x)
{
    return x>=0?1.0:-1.0;
}
void _AttractionFieldMap(int n_lines, const float* lines,
                         int height, int width, float* afm,int* label)
{
    for(int h = 0; h < height; ++h){
        for(int w  = 0; w < width; ++w){
            float min_dis = __FLT_MAX__;
            float ax_opt  = 0;
            float ay_opt  = 0;
            int  ind_opt  = 0;
            
            float px = (float) w;
            float py = (float) h;

            for(int i = 0; i < n_lines; ++i){
                float dx = lines[4*i+2] - lines[4*i];
                float dy = lines[4*i+3] - lines[4*i+1];
                float norm2 = dx*dx + dy*dy;

                float t = ((px-lines[4*i])*dx + (py-lines[4*i+1])*dy)/(norm2+1e-6);

                t = t<1.0?t:1.0;
                t = t>0.0?t:0.0;

                float ax = lines[4*i] + t*dx - px;
                float ay = lines[4*i+1] + t*dy - py;

                float dis = ax*ax + ay*ay;
                if (dis < min_dis){
                    min_dis = dis;
                    ax_opt = ax;
                    ay_opt = ay;
                    ind_opt = i;
                }
            }

            afm[h*width + w] = -sgn(ax_opt)*log(fabs(ax_opt/float(width))+1e-6);
            afm[h*width + w + height*width] = -sgn(ay_opt)*log(fabs(ay_opt/float(height))+1e-6);
            label[h*width + w] = ind_opt;
        }
    }
}