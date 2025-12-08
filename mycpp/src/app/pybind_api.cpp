/*
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
*/


#include "Utils.h"
#include <boost/algorithm/string.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

namespace py = pybind11;



//@angle_diff: unit is degree
//@dist_diff: unit is meter
vectorMatrix4f cluster_poses(float angle_diff, float dist_diff, const vectorMatrix4f &poses_in, const vectorMatrix4f &symmetry_tfs)
{
  printf("num original candidates = %d\n",poses_in.size());
  vectorMatrix4f poses_out;
  poses_out.push_back(poses_in[0]);

  const float radian_thres = angle_diff/180.0*M_PI;

  for (int i=1;i<poses_in.size();i++)
  {
    bool isnew = true;
    Eigen::Matrix4f cur_pose = poses_in[i];
    for (const auto &cluster:poses_out)
    {
      Eigen::Vector3f t0 = cluster.block(0,3,3,1);
      Eigen::Vector3f t1 = cur_pose.block(0,3,3,1);

      if ((t0-t1).norm()>=dist_diff)
      {
        continue;
      }

      for (const auto &tf: symmetry_tfs)
      {
        Eigen::Matrix4f cur_pose_tmp = cur_pose*tf;
        float rot_diff = Utils::rotationGeodesicDistance(cur_pose_tmp.block(0,0,3,3), cluster.block(0,0,3,3));
        if (rot_diff < radian_thres)
        {
          isnew = false;
          break;
        }
      }

      if (!isnew) break;
    }

    if (isnew)
    {
      poses_out.push_back(poses_in[i]);
    }
  }

  printf("num of pose after clustering: %d\n",poses_out.size());
  return poses_out;
}





// Helper function to convert numpy array to vectorMatrix4f
vectorMatrix4f numpy_to_vector(py::array_t<float> arr) {
  printf("numpy_to_vector: start\n");
  py::buffer_info buf = arr.request();
  printf("numpy_to_vector: got buffer info, ndim=%ld\n", buf.ndim);

  if (buf.ndim != 3 || buf.shape[1] != 4 || buf.shape[2] != 4) {
    throw std::runtime_error("Input must be Nx4x4 array");
  }

  vectorMatrix4f result;
  float* ptr = static_cast<float*>(buf.ptr);
  size_t n = buf.shape[0];
  printf("numpy_to_vector: converting %zu matrices\n", n);

  for (size_t i = 0; i < n; i++) {
    Eigen::Matrix4f mat;
    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 4; c++) {
        mat(r, c) = ptr[i * 16 + r * 4 + c];
      }
    }
    result.push_back(mat);
  }
  printf("numpy_to_vector: done\n");
  return result;
}

// Helper function to convert vectorMatrix4f to numpy array
py::array_t<float> vector_to_numpy(const vectorMatrix4f& vec) {
  printf("vector_to_numpy: start with %zu matrices\n", vec.size());
  size_t n = vec.size();
  std::vector<size_t> shape = {n, 4, 4};
  py::array_t<float> result(shape);
  auto buf = result.request();
  float* ptr = static_cast<float*>(buf.ptr);

  for (size_t i = 0; i < n; i++) {
    for (int r = 0; r < 4; r++) {
      for (int c = 0; c < 4; c++) {
        ptr[i * 16 + r * 4 + c] = vec[i](r, c);
      }
    }
  }
  printf("vector_to_numpy: done\n");
  return result;
}

// Wrapper that accepts numpy arrays
py::array_t<float> cluster_poses_numpy(float angle_diff, float dist_diff,
                                       py::array_t<float> poses_in_np,
                                       py::array_t<float> symmetry_tfs_np) {
  printf("cluster_poses_numpy: converting inputs\n");
  vectorMatrix4f poses_in = numpy_to_vector(poses_in_np);
  vectorMatrix4f symmetry_tfs = numpy_to_vector(symmetry_tfs_np);
  printf("cluster_poses_numpy: calling cluster_poses\n");
  vectorMatrix4f result = cluster_poses(angle_diff, dist_diff, poses_in, symmetry_tfs);
  printf("cluster_poses_numpy: converting output\n");
  return vector_to_numpy(result);
}

PYBIND11_MODULE(mycpp, m)
{
  m.def("cluster_poses", &cluster_poses_numpy);
}