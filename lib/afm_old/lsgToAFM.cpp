#include "lsgToAFM.hpp"
#include <iostream>
#include <math.h>
#include <stdio.h>
void AFMFromLSG(const float* lsg, const float* point, float* output)
{
    float dx = lsg[2]-lsg[0]; // (s2-s1)_x
    float dy = lsg[3]-lsg[1]; // (s2-s1)_y
    float norm2 = dx*dx + dy*dy;

    float t = ((point[0]-lsg[0])*dx + (point[1]-lsg[1])*dy)/(norm2+1e-6);
//    printf("%f\n",t);
    t = t<1.0? t: 1.0;
    t = t>0.0? t: 0.0;

    output[0] = lsg[0] + t*(lsg[2]-lsg[0]) - point[0];
    output[1] = lsg[1] + t*(lsg[3]-lsg[1]) - point[1];
}

float sgn(float x)
{
    return x>=0?1.0:-1.0;
}
void _lsgToAFM(int input_H, int input_W,
                     int lsg_num, const float* lsgs,
                     int output_H, int output_W, float* offset,
                     int* labels)
{
//    printf("%d\n", lsg_num);
    for(int y = 0; y < output_H; ++y)
    {
        for(int x = 0; x < output_W; ++x)
        {
            float lsg_normalized[4];
            float point[2] = {(float) x, (float) y};
            float mindis = 1e20;
            float out_point[2];
            for(int i = 0; i < lsg_num; ++i){
                lsg_normalized[0] = lsgs[4*i] * (float) output_W / (float) input_W ;
                lsg_normalized[1] = lsgs[4*i + 1]  * (float) output_H / (float) input_H;
                lsg_normalized[2] = lsgs[4*i + 2]  * (float) output_W / (float) input_W;
                lsg_normalized[3] = lsgs[4*i + 3]  * (float) output_H / (float) input_H;
                AFMFromLSG(lsg_normalized, point, out_point);
                float dis = out_point[0]*out_point[0]+out_point[1]*out_point[1];
//                printf("%f %f\n",out_point[0],out_point[1]);
                if (dis<mindis) {
                    mindis = dis;
                    offset[y*output_W + x] = -sgn(out_point[0])*log(fabs(out_point[0]/(float)output_W)+1e-6);
                    offset[y*output_W + x + output_H* output_W] = -sgn(out_point[1])*log(fabs(out_point[1]/(float)output_H)+1e-6);
                    labels[y*output_W + x] = i;
                }
            }
        }
    }
}
