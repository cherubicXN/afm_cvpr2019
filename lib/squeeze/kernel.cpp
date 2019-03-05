#include "squeeze.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/** 3/2 pi*/
#define M_3_2_PI 4.71238898038
/** 2 pi */
#define M_2__PI 6.28318530718


class Point {
public:
    Point(){}
    Point(float x, float y, float ang) :
        x_(x), y_(y), ang_(ang), used_(false) {}
    inline float x() const { return this->x_; }
    inline float y() const { return this->y_; }
    inline float ang() const { return this->ang_;}
    inline bool used() const { return this->used_;}
    inline float &x() { return this->x_;}
    inline float &y() { return this->y_;}
    inline float &ang() { return this->ang_;}
    inline bool &used() { return this->used_;}
private:
    float x_,y_, ang_;
    bool used_;
};

class Rectangle
{
public:
    float x1, y1, x2, y2;
    float width;
    float l_min, l_max, w_min, w_max;
    float x,y;
    float theta;
    float dx,dy;
    float prec;
    float p;
};

class PoLsMap {
public:
    struct PoLs {
        PoLs() {inds.clear(); }
        ~PoLs() {inds.clear(); }
        inline void push_back(int id) {inds.push_back(id); }
        inline size_t size() { return inds.size(); }
        std::vector<int> inds;
    };

    PoLsMap(int height, int width, const float *arr_x, const float *arr_y, const float *arr_ang, int cnt)
    : height_(height), width_(width)
    {
        pols_ = new PoLs[height*width];
        for(int i = 0; i < cnt; ++i)
        {
            int x_int = int(arr_x[i]);
            int y_int = int(arr_y[i]);
            points_.push_back(Point(arr_x[i],arr_y[i],arr_ang[i]));
            pols_[y_int*width_ + x_int].push_back(i);
        }
    }
    const PoLs& operator () (int x, int y) const
    {
        return pols_[y*width_ + x];
    }
    PoLs& operator() (int x, int y)
    {
        return pols_[y*width_ + x];
    }

    const Point& operator () (int x, int y, int id) const
    {
        return points_[pols_[y*width_+x].inds[id]];
    }
    Point& operator () (int x, int y, int id)
    {
        return points_[pols_[y*width_+x].inds[id]];
    }

    int height_, width_;
    PoLs* pols_;
    std::vector<Point> points_;

};
bool isaligned(float phi, float theta, float prec)
{
    theta -= phi;
    if (theta < 0.0) theta = - theta;
    if (theta > M_3_2_PI)
    {
        theta -= M_2__PI;
        if (theta < 0.0) theta = -theta;
    }

    return theta <=prec;
}

float angle_diff(float a, float b)
{
    a -= b;
    while( a <= -M_PI ) a += M_2__PI;
    while( a >   M_PI ) a -= M_2__PI;
    if( a < 0.0 ) a = -a;
    return a;
}
void region_grow(int x, int y, float ang,
                float &reg_ang,
                std::vector<Point>& reg,
                std::vector<Point>& reg_int,
                std::vector<float>& confidence,
                PoLsMap& map,
                float prec)
{
    reg.clear();
    reg_int.clear();
    confidence.clear();
    if (map(x,y).size() == 0)
        return;
    reg_ang = ang;
    float sumdx = 0;
    float sumdy = 0;
    int cnt = 0;
    for(int id = 0; id < map(x,y).size(); ++id) {
        Point& pt = map(x,y,id);
        if (pt.used())
            continue;
        if (isaligned(reg_ang,pt.ang(),prec))
        {
            pt.used() = true;
            reg.push_back(pt);

            sumdx += cos(pt.ang());
            sumdy += sin(pt.ang());
            reg_ang = atan2(sumdy, sumdx);
            ++cnt;
        }
    }

    if (reg.size() == 0)
        return;
//    std::vector<Point> reg_int;

    reg_int.push_back(Point(int(x),int(y),reg_ang));
    confidence.push_back((float)cnt/(float)map(x,y).size());
    for(int i = 0; i < reg_int.size(); ++i)
    {
        for(int xx = (int)reg_int[i].x()-1; xx<= (int)reg_int[i].x()+1; ++xx)
        {
            for(int yy = (int)reg_int[i].y()-1; yy<= (int)reg_int[i].y()+1; ++yy)
            {
                if (xx<0 || yy<0 || xx >= map.width_ || yy >= map.height_ || map(xx,yy).size() <=1)
                    continue;
                bool flag = false;
                cnt = 0;
                for(int k = 0; k < map(xx,yy).size(); ++k) {
                    Point& pt = map(xx,yy,k);
                    if(isaligned(reg_ang, pt.ang(), prec)&&pt.used()==false)
                    {
                        reg.push_back(pt);
                        pt.used() = true;
                        flag = true;
                        sumdx += cos(pt.ang());
                        sumdy += sin(pt.ang());
                        reg_ang = atan2(sumdy, sumdx);
                        ++cnt;
                    }
                }
                if (flag == false)
                    continue;
                // for(int k = 0; k < map(xx,yy).size(); ++k) {
                //     Point& pt = map(xx,yy,k);
                //     pt.used() = true;
                // }


                reg_int.push_back(Point(xx,yy,reg_ang));
                confidence.push_back((float)cnt/(float)map(xx,yy).size());
            }
        }
    }
}


bool region2rect(const std::vector<Point> &reg_int,
                 const std::vector<float> &confidence,
                 float reg_angle, float prec, float p, Rectangle &rect)
{
    float x,y,dx,dy,l,w,theta,weight,sum,l_min,l_max,w_min,w_max;
    x = y = sum = 0.0;
    for(int i=0; i<reg_int.size(); ++i)
    {
        weight = confidence[i];
        // weight = 1.0;
        x += reg_int[i].x()*weight;
        y += reg_int[i].y()*weight;
        sum += weight;
    }
    if (sum<1e-6)
        return false;
    x /= sum;
    y /= sum;

    // theta = get_theta(reg_int, confidence,reg_angle, prec, x,y);
    theta = reg_angle;

    dx = cos(theta);
    dy = sin(theta);
    l_min = l_max = w_max = w_min = 0.0;
    for(int i=0; i<reg_int.size(); ++i)
    {
        l = (reg_int[i].x() - x)*dx + (reg_int[i].y() - y)*dy;
        w = -(reg_int[i].x() - x)*dy + (reg_int[i].y() - y)*dx;
      if( l > l_max ) l_max = l;
      if( l < l_min ) l_min = l;
      if( w > w_max ) w_max = w;
      if( w < w_min ) w_min = w;
    }

    rect.x1 = x + l_min*dx;
    rect.y1 = y + l_min*dy;
    rect.x2 = x + l_max*dx;
    rect.y2 = y + l_max*dy;
    rect.l_min = l_min;
    rect.l_max = l_max;
    rect.w_min = w_min;
    rect.w_max = w_max;
    rect.width = w_max - w_min;
    rect.x = x;
    rect.y = y;
    rect.theta = theta;
    rect.dx = dx;
    rect.dy = dy;
    rect.prec = prec;
    rect.p = p;
    if (rect.width < 1.0)
        rect.width = 1.0;

    // return rect.width/(rect.l_max-rect.l_min)<0.3;
    return true;
}

// (int x, int y, float ang,
//                 float &reg_ang,
//                 std::vector<Point>& reg,
//                 std::vector<Point>& reg_int,
//                 std::vector<float>& confidence,
//                 PoLsMap& map,
//                 float prec)
void refine(std::vector<Point> &reg_int,
            Rectangle& rect, std::vector<float> &confidence, PoLsMap& map)
{
    
    float xc = rect.x;
    float yc = rect.y;
    float ang_rect = rect.theta;    
    float dx  = cos(ang_rect), dy = sin(ang_rect);
    
    float x1 = rect.l_min*dx - rect.w_min*dy;
    float y1 = rect.l_min*dy + rect.w_min*dx;
    float x2 = rect.l_max*dx - rect.w_min*dy;
    float y2 = rect.l_max*dy + rect.w_min*dx;
    float x3 = rect.l_max*dx - rect.w_max*dy;
    float y3 = rect.l_max*dy + rect.w_max*dx;
    float x4 = rect.l_min*dx - rect.w_max*dy;
    float y4 = rect.l_min*dy + rect.w_max*dx;

    std::vector<Point> reg_rot;
    for(int i = 0; i < reg_int.size(); ++i)
    {
        Point p;
        p.x() = dx*(reg_int[i].x()-xc) + dy*(reg_int[i].y()-yc);
        p.y() = -dy*(reg_int[i].x()-xc) + dx*(reg_int[i].y()-yc);
        reg_rot.push_back(p);
        // std::cout<<reg_int[i].x()<<" "<<reg_int[i].y()<<" "<<p.x()<<" "<<p.y()<<"\n";
    }
    int start_w = (int) floor(rect.w_min);
    int end_w = (int) ceil(rect.w_max);
    int start_l = (int) floor(rect.l_min);
    int end_l = (int) ceil(rect.l_max);
    std::vector< std::vector<int> > indexes(end_w-start_w);

    for(int i = 0; i < reg_rot.size(); ++i)
    {
        const Point &p = reg_rot[i];
        if (p.x()<start_l || p.x()>=end_l || p.y()<start_w || p.y()>=end_w){
            continue;
        }
        for(int k = start_w; k < end_w; ++k)
        {
            if(p.y()>=k && p.y()<k+1)
            {
                indexes[k-start_w].push_back(i);
                break;
            }
        }
    }
    for(int k = start_w; k < end_w; ++k)
    {
        float ratio = (float)(indexes[k-start_w].size()/(end_l-start_l));
        const std::vector<int>& local_ind = indexes[k-start_w];
        if(ratio<=0.1)
            continue;
        for(int i = 0; i < local_ind.size();++i)
        {
            int id = local_ind[i];
            int x = reg_int[id].x();
            int y = reg_int[id].y();
            for(int n = 0; n < map(x,y).size();++n)
                map(x,y,n).used() = false;
        }        
        // std::cout<<indexes[k-start_w].size()<<"/"<<end_l-start_l<<"\n";
    }
    // int xx = reg_int[0].x();
    // int yy = reg_int[0].y();
    // std::vector<Point> reg_new;
    // confidence.clear();
    // reg_int.clear();
    // region_grow(xx,yy,ang_rect,ang_rect,reg_new,reg_int,confidence,map,10.0/180.0*M_PI);
}


void _region_grow(int H, int W,
                const float* arr_x, const float* arr_y, const float* arr_ang,int cnt,
                float* rectangles, int *num)
{
    float prec = 10.0/180.0*M_PI; //rad = degree/180*pi
    float p = prec/M_PI;

    PoLsMap map(H, W, arr_x, arr_y, arr_ang, cnt);

    std::vector<Rectangle> vec_rects;    
    for(int i = 0; i < cnt; ++i)
    {
        std::vector<Point> reg, reg_int;
        std::vector<float> confidence;
        float reg_ang = 0;
        region_grow(arr_x[i],arr_y[i],arr_ang[i],reg_ang,reg,reg_int,confidence,map,prec);
        if(reg_int.size() <= 5)
            continue;
        Rectangle rect;
        if(!region2rect(reg_int, confidence, reg_ang, prec, p, rect ))
            continue;
        vec_rects.push_back(rect);
    }
    *num = vec_rects.size();

    for(int i = 0; i < *num; ++i)
    {
        // x1,y1,x2,y2,ratio
        // ratio = width/length
        rectangles[5*i + 0] = vec_rects[i].x1;
        rectangles[5*i + 1] = vec_rects[i].y1;
        rectangles[5*i + 2] = vec_rects[i].x2;
        rectangles[5*i + 3] = vec_rects[i].y2;
        float length = sqrt((vec_rects[i].x1-vec_rects[i].x2)*(vec_rects[i].x1-vec_rects[i].x2) +
                            (vec_rects[i].y1-vec_rects[i].y2)*(vec_rects[i].y1-vec_rects[i].y2));
        rectangles[5*i + 4] = vec_rects[i].width;
    }
}