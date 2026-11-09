/* Independent driver for the public ISO/IEC 21122-5 edition-3 software.
 * Compile with libjxs/src/dwt.c and the archive's include paths.
 * Contains no copied reference implementation. */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "dwt.h"

/* ids_xy_val_t is signed four-bit, so level 8 cannot be represented in
 * one reference call. Compose one level with a seven-level LL call instead. */
static int transform(int32_t *a,int w,int h,int nx,int ny,int inverse) {
    ids_t ids={0}; xs_image_t im={0};
    ids.ncomps=1; ids.comp_w[0]=w; ids.comp_h[0]=h; im.comps_array[0]=a;
    ids.nlxyp[0].x=nx<=7?nx:1;
    ids.nlxyp[0].y=nx<=7?ny:(ny>0);
    if(nx<=7) {
        if(inverse) dwt_inverse_transform(&ids,&im); else dwt_forward_transform(&ids,&im);
        return 0;
    }
    if(!inverse) dwt_forward_transform(&ids,&im);
    int lw=(w+1)/2,lh=ny?(h+1)/2:h,sy=ny?2:1;
    int32_t *low=calloc((size_t)lw*lh,sizeof(*low)); if(!low) return 2;
    for(int y=0;y<lh;++y) for(int x=0;x<lw;++x) low[y*lw+x]=a[y*sy*w+x*2];
    int r=transform(low,lw,lh,nx-1,ny?ny-1:0,inverse);
    if(r) { free(low); return r; }
    for(int y=0;y<lh;++y) for(int x=0;x<lw;++x) a[y*sy*w+x*2]=low[y*lw+x];
    free(low);
    if(inverse) dwt_inverse_transform(&ids,&im);
    return 0;
}

static int cell(int w,int h,int nx,int ny,int depth,int fq) {
    int32_t *a=calloc((size_t)w*h,sizeof(*a));
    if (!a) return 2;
    for(int i=0;i<w*h;++i) {
        uint32_t k=(uint32_t)i;
        uint32_t sample=(k*73u+k*k*19u+17u)&((1u<<depth)-1);
        a[i]=((int32_t)sample-(1<<(depth-1))) * (1<<fq);
    }
    int r=transform(a,w,h,nx,ny,0); if(r) return r;
    uint64_t hash=UINT64_C(14695981039346656037);
    for(int i=0;i<w*h;++i) for(int j=0;j<4;++j) {
        hash ^= ((uint32_t)a[i] >> (8*j)) & 255;
        hash *= UINT64_C(1099511628211);
    }
    printf("%d %d %d %d %d %d %016" PRIx64 "\n",w,h,nx,ny,depth,fq,hash);
    r=transform(a,w,h,nx,ny,1); if(r) return r;
    for(int i=0;i<w*h;++i) {
        uint32_t k=(uint32_t)i;
        uint32_t sample=(k*73u+k*k*19u+17u)&((1u<<depth)-1);
        if(a[i]!=((int32_t)sample-(1<<(depth-1)))*(1<<fq)) return 3;
    }
    free(a); return 0;
}

int main(void) {
    const int shapes[][4]={{2,2,1,1},{3,3,1,1},{4,5,2,2},{5,4,2,2},
        {8,9,3,2},{9,8,3,2},{32,4,5,2},{33,5,5,2},{32,2,5,0},{33,3,5,0}};
    for(unsigned s=0;s<sizeof(shapes)/sizeof(shapes[0]);++s)
    for(int depth=8;depth<=16;++depth) for(int fi=0;fi<3;++fi) {
        int fq=fi==0?0:(fi==1?6:8);
        int r=cell(shapes[s][0],shapes[s][1],shapes[s][2],shapes[s][3],depth,fq);
        if(r) return r;
    }
    /* Numerical cascade coverage, including peer-supported deeper levels.
     * These cells do not assert named-profile or header conformance. */
    for(int nx=1;nx<=8;++nx) for(int ny=0;ny<=nx;++ny)
    for(int odd=0;odd<=1;++odd) for(int depth=8;depth<=16;depth+=8)
    for(int fi=0;fi<3;++fi) {
        int fq=fi==0?0:(fi==1?6:8);
        int r=cell((1<<nx)+odd,(ny?(1<<ny):2)+odd,nx,ny,depth,fq);
        if(r) return r;
    }
    return 0;
}
