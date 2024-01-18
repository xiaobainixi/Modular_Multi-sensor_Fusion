#pragma once

#include <sophus/se3.hpp>
#include <sophus/sim3.hpp>

#include "eigen_utils.hpp"

namespace Sophus {

template <typename Scalar>
inline typename SE3<Scalar>::Tangent se3_logd(const SE3<Scalar> &se3) {
  typename SE3<Scalar>::Tangent upsilon_omega;
  upsilon_omega.template tail<3>() = se3.so3().log();
  upsilon_omega.template head<3>() = se3.translation();

  return upsilon_omega;
}

template <typename Derived>
inline SE3<typename Derived::Scalar> se3_expd(
    const Eigen::MatrixBase<Derived> &upsilon_omega) {
  using Scalar = typename Derived::Scalar;

  return SE3<Scalar>(SO3<Scalar>::exp(upsilon_omega.template tail<3>()),
                     upsilon_omega.template head<3>());
}

template <typename Scalar>
inline typename Sim3<Scalar>::Tangent sim3_logd(const Sim3<Scalar> &sim3) {
  typename Sim3<Scalar>::Tangent upsilon_omega_sigma;
  upsilon_omega_sigma.template tail<4>() = sim3.rxso3().log();
  upsilon_omega_sigma.template head<3>() = sim3.translation();

  return upsilon_omega_sigma;
}


template <typename Derived>
inline Sim3<typename Derived::Scalar> sim3_expd(
    const Eigen::MatrixBase<Derived> &upsilon_omega_sigma) {
  using Scalar = typename Derived::Scalar;

  return Sim3<Scalar>(
      RxSO3<Scalar>::exp(upsilon_omega_sigma.template tail<4>()),
      upsilon_omega_sigma.template head<3>());
}


template <typename Derived1, typename Derived2>
inline void rightJacobianSO3(const Eigen::MatrixBase<Derived1> &phi,
                             const Eigen::MatrixBase<Derived2> &J_phi) {

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

  Scalar phi_norm2 = phi.squaredNorm();
  Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
  Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

  J.setIdentity();

  if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
    Scalar phi_norm = std::sqrt(phi_norm2);
    Scalar phi_norm3 = phi_norm2 * phi_norm;

    J -= phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
    J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
  } else {
    // Taylor expansion around 0
    J -= phi_hat / 2;
    J += phi_hat2 / 6;
  }
}

template <typename Derived1, typename Derived2>
inline void rightJacobianInvSO3(const Eigen::MatrixBase<Derived1> &phi,
                                const Eigen::MatrixBase<Derived2> &J_phi) {

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

  Scalar phi_norm2 = phi.squaredNorm();
  Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
  Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

  J.setIdentity();
  J += phi_hat / 2;

  if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
    Scalar phi_norm = std::sqrt(phi_norm2);

    // We require that the angle is in range [0, pi]. We check if we are close
    // to pi and apply a Taylor expansion to scalar multiplier of phi_hat2.
    // Technically, log(exp(phi)exp(epsilon)) is not continuous / differentiable
    // at phi=pi, but we still aim to return a reasonable value for all valid
    // inputs.

    if (phi_norm < M_PI - Sophus::Constants<Scalar>::epsilonSqrt()) {
      // regular case for range (0,pi)
      J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) /
                                           (2 * phi_norm * std::sin(phi_norm)));
    } else {
      // 0th-order Taylor expansion around pi
      J += phi_hat2 / (M_PI * M_PI);
    }
  } else {
    // Taylor expansion around 0
    J += phi_hat2 / 12;
  }
}

template <typename Derived1, typename Derived2>
inline void leftJacobianSO3(const Eigen::MatrixBase<Derived1> &phi,
                            const Eigen::MatrixBase<Derived2> &J_phi) {

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

  Scalar phi_norm2 = phi.squaredNorm();
  Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
  Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

  J.setIdentity();

  if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
    Scalar phi_norm = std::sqrt(phi_norm2);
    Scalar phi_norm3 = phi_norm2 * phi_norm;

    J += phi_hat * (1 - std::cos(phi_norm)) / phi_norm2;
    J += phi_hat2 * (phi_norm - std::sin(phi_norm)) / phi_norm3;
  } else {
    // Taylor expansion around 0
    J += phi_hat / 2;
    J += phi_hat2 / 6;
  }
}

template <typename Derived1, typename Derived2>
inline void leftJacobianInvSO3(const Eigen::MatrixBase<Derived1> &phi,
                               const Eigen::MatrixBase<Derived2> &J_phi) {

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

  Scalar phi_norm2 = phi.squaredNorm();
  Eigen::Matrix<Scalar, 3, 3> phi_hat = Sophus::SO3<Scalar>::hat(phi);
  Eigen::Matrix<Scalar, 3, 3> phi_hat2 = phi_hat * phi_hat;

  J.setIdentity();
  J -= phi_hat / 2;

  if (phi_norm2 > Sophus::Constants<Scalar>::epsilon()) {
    Scalar phi_norm = std::sqrt(phi_norm2);

    // We require that the angle is in range [0, pi]. We check if we are close
    // to pi and apply a Taylor expansion to scalar multiplier of phi_hat2.
    // Technically, log(exp(phi)exp(epsilon)) is not continuous / differentiable
    // at phi=pi, but we still aim to return a reasonable value for all valid
    // inputs.

    if (phi_norm < M_PI - Sophus::Constants<Scalar>::epsilonSqrt()) {
      // regular case for range (0,pi)
      J += phi_hat2 * (1 / phi_norm2 - (1 + std::cos(phi_norm)) /
                                           (2 * phi_norm * std::sin(phi_norm)));
    } else {
      // 0th-order Taylor expansion around pi
      J += phi_hat2 / (M_PI * M_PI);
    }
  } else {
    // Taylor expansion around 0
    J += phi_hat2 / 12;
  }
}

template <typename Derived1, typename Derived2>
inline void rightJacobianSE3Decoupled(
    const Eigen::MatrixBase<Derived1> &phi,
    const Eigen::MatrixBase<Derived2> &J_phi) {

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

  J.setZero();

  Eigen::Matrix<Scalar, 3, 1> omega = phi.template tail<3>();
  rightJacobianSO3(omega, J.template bottomRightCorner<3, 3>());
  J.template topLeftCorner<3, 3>() =
      Sophus::SO3<Scalar>::exp(omega).inverse().matrix();
}

template <typename Derived1, typename Derived2>
inline void rightJacobianInvSE3Decoupled(
    const Eigen::MatrixBase<Derived1> &phi,
    const Eigen::MatrixBase<Derived2> &J_phi) {

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

  J.setZero();

  Eigen::Matrix<Scalar, 3, 1> omega = phi.template tail<3>();
  rightJacobianInvSO3(omega, J.template bottomRightCorner<3, 3>());
  J.template topLeftCorner<3, 3>() = Sophus::SO3<Scalar>::exp(omega).matrix();
}

template <typename Derived1, typename Derived2>
inline void rightJacobianSim3Decoupled(
    const Eigen::MatrixBase<Derived1> &phi,
    const Eigen::MatrixBase<Derived2> &J_phi) {

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

  J.setZero();

  Eigen::Matrix<Scalar, 4, 1> omega = phi.template tail<4>();
  rightJacobianSO3(omega.template head<3>(), J.template block<3, 3>(3, 3));
  J.template topLeftCorner<3, 3>() =
      Sophus::RxSO3<Scalar>::exp(omega).inverse().matrix();
  J(6, 6) = 1;
}

template <typename Derived1, typename Derived2>
inline void rightJacobianInvSim3Decoupled(
    const Eigen::MatrixBase<Derived1> &phi,
    const Eigen::MatrixBase<Derived2> &J_phi) {

  using Scalar = typename Derived1::Scalar;

  Eigen::MatrixBase<Derived2> &J =
      const_cast<Eigen::MatrixBase<Derived2> &>(J_phi);

  J.setZero();

  Eigen::Matrix<Scalar, 4, 1> omega = phi.template tail<4>();
  rightJacobianInvSO3(omega.template head<3>(), J.template block<3, 3>(3, 3));
  J.template topLeftCorner<3, 3>() = Sophus::RxSO3<Scalar>::exp(omega).matrix();
  J(6, 6) = 1;
}

}  // namespace Sophus
