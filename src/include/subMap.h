// Author of  EPSC-LOAM : QZ Wang
// Email wangqingzhi27@outlook.com

#ifndef _SUBMAP_H_
#define _SUBMAP_H_

#include "utility.h"

struct submap_t {
  void append_feature(const keyframe_t &in_cblock, bool append_down) {
    // pc_raw->points.insert(pc_raw->points.end(),
    // in_cblock.pc_raw->points.begin(), in_cblock.pc_raw->points.end());
    if (!append_down) {
      // if (used_feature_type[0] == '1')
      // 	pc_ground->points.insert(pc_ground->points.end(),
      // in_cblock.pc_ground->points.begin(),
      // in_cblock.pc_ground->points.end());

    } else {
    }
  }
};

#endif  // _SUBMAP_H_