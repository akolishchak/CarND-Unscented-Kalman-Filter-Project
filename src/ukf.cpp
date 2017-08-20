#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>
#include "tools.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF()
{
    is_initialized_ = false;

    //set state dimension
    n_x_ = 5;

    // initial state vector
    x_ = VectorXd(n_x_);

    // initial covariance matrix
    P_ = MatrixXd::Identity(n_x_, n_x_);

    //set augmented dimension
    n_aug_ = 7;

    // set number of sigma points
    n_sig_ = 2 * n_aug_ + 1;

    // Predicted sigma points
    Xsig_pred_ = MatrixXd(n_x_, n_sig_);

    //define spreading parameter
    lambda_ = 3 - n_aug_;

    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1; // 22mph per 20 sec ~ 0.5 m/s^2

    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 1;

    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;

    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;

    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;

    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;

    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;

    //set measurement dimension, radar can measure r, phi, and r_dot
    n_z_radar_ = 3;

    // set radar R matrix
    R_radar_ = MatrixXd(n_z_radar_, n_z_radar_);
    R_radar_ << std_radr_ * std_radr_, 0, 0,
                0, std_radphi_ * std_radphi_, 0,
                0, 0, std_radrd_ * std_radrd_;

    // set measurement dimension, lidar can measure px and py
    n_z_laser_ = 2;

    // set lidar R matrix
    R_laser_ = MatrixXd(n_z_laser_, n_z_laser_);
    R_laser_ << std_laspx_ * std_laspx_, 0,
                0, std_laspy_ * std_laspy_;

    //create vector for weights
    weights_ = VectorXd(n_sig_);

    // set weights
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    double weight_i = 0.5 / (n_aug_ + lambda_);
    for (int i=1; i < n_sig_; i++) {  //n_sig_ weights
        weights_(i) = weight_i;
    }
}

UKF::~UKF()
{

}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package)
{
    /*****************************************************************************
     *  Initialization
     ****************************************************************************/
    if (!is_initialized_) {
        /**
          * Initialize the state ekf_.x_ with the first measurement.
          * Create the covariance matrix.
          * Remember: you'll need to convert radar from polar to cartesian coordinates.
        */
        // first measurement
        float px = 0, py = 0, vx = 0, vy = 0;

        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            /**
            Convert radar from polar to cartesian coordinates and initialize state.
            */
            float rho = meas_package.raw_measurements_[0];
            float phi = meas_package.raw_measurements_[1];
            float rho_dot = meas_package.raw_measurements_[2];
            px = rho * cos(phi);
            py = rho * sin(phi);
            vx = rho_dot * cos(phi);
            vy = rho_dot * sin(phi);
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            /**
            Initialize state.
            */
            //set the state with the initial location and zero velocity
            px = meas_package.raw_measurements_[0];
            py = meas_package.raw_measurements_[1];
        }

        x_ << px, py, sqrt(vx*vx + vy*vy), 0, 0;

        previous_timestamp_ = meas_package.timestamp_;

        // done initializing, no need to predict or update
        is_initialized_ = true;
        return;
    }

    //
    // Prediction
    //

    //compute the time elapsed between the current and previous measurements
    double delta_t = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
    previous_timestamp_ = meas_package.timestamp_;

    Prediction(delta_t);

    //
    // Update
    //
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
        UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
        UpdateLidar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t)
{
    //
    // 1. Create augmented state and sigma points
    //
    //create augmented mean vector
    VectorXd x_aug = VectorXd::Zero(n_aug_);
    x_aug.head(n_x_) = x_;

    //create augmented covariance matrix
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    P_aug.topLeftCorner(n_x_, n_x_) = P_;
    P_aug(n_x_, n_x_) = std_a_ * std_a_;
    P_aug(n_x_+1, n_x_+1) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
    for (unsigned int i = 0; i < Xsig_aug.cols(); i++)
        Xsig_aug.col(i) = x_aug;

    MatrixXd Lk = sqrt(lambda_ + n_aug_)*L;
    Xsig_aug.block(0, 1, n_aug_, n_aug_) += Lk;
    Xsig_aug.block(0, n_aug_+1, n_aug_, n_aug_) -= Lk;

    //
    // 2. Predict sigma points
    //
    for (int i = 0; i< n_sig_; i++)
    {
        //extract values for better readability
        double p_x = Xsig_aug(0, i);
        double p_y = Xsig_aug(1, i);
        double v = Xsig_aug(2, i);
        double yaw = Xsig_aug(3, i);
        double yawd = Xsig_aug(4, i);
        double nu_a = Xsig_aug(5, i);
        double nu_yawdd = Xsig_aug(6, i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (fabs(yawd) > min_val) {
            px_p = p_x + v / yawd * (sin(yaw + yawd * delta_t) - sin(yaw));
            py_p = p_y + v / yawd * (cos(yaw) - cos(yaw + yawd * delta_t));
        }
        else {
            px_p = p_x + v * delta_t * cos(yaw);
            py_p = p_y + v * delta_t * sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd * delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5 * nu_a * delta_t * delta_t * cos(yaw);
        py_p = py_p + 0.5 * nu_a * delta_t * delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;

        yaw_p = yaw_p + 0.5 * nu_yawdd * delta_t * delta_t;
        yawd_p = yawd_p + nu_yawdd * delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0, i) = px_p;
        Xsig_pred_(1, i) = py_p;
        Xsig_pred_(2, i) = v_p;
        Xsig_pred_(3, i) = yaw_p;
        Xsig_pred_(4, i) = yawd_p;
    }

    //
    // 3. Predict the state and the state covariance
    //
    //predicted state mean
    x_.fill(0.0);
    for (int i = 0; i < n_sig_; i++) {  //iterate over sigma points
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < n_sig_; i++) {  //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        Tools::NormalizeAngle(x_diff(3));

        P_ += weights_(i) * x_diff * x_diff.transpose();
    }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package)
{
    //
    // 1. Predict laser measurement
    //
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z_laser_, n_sig_);

    //transform sigma points into measurement space
    for (int i = 0; i < n_sig_; i++) {  //n_sig_ simga points
        // extract values for better readability
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);

        // measurement model
        Zsig(0, i) = p_x;
        Zsig(1, i) = p_y;
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd::Zero(n_z_laser_);
    for (int i=0; i < n_sig_; i++) {
        z_pred += weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z_laser_, n_z_laser_);
    for (int i = 0; i < n_sig_; i++) {  //n_sig_ simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    S = S + R_laser_;

    //
    // 2. Update the state and the state covariance matrix
    //
    // set measurement vector
    VectorXd z = VectorXd(n_z_laser_);
    for (unsigned int i = 0; i < n_z_laser_; i++)
        z(i) = meas_package.raw_measurements_[i];

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_laser_);
    for (int i = 0; i < n_sig_; i++) {  //n_sig_ simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //update state mean and covariance matrix
    double NIS = UpdateState(K, z_diff, S);
    cout << "NIS(laser) = " << NIS << endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package)
{
    //
    // 1. Predict radar measurement
    //
    //create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z_radar_, n_sig_);

    //transform sigma points into measurement space
    for (int i = 0; i < n_sig_; i++) {  //n_sig_ simga points
        // extract values for better readability
        double p_x = Xsig_pred_(0, i);
        double p_y = Xsig_pred_(1, i);
        double v  = Xsig_pred_(2, i);
        double yaw = Xsig_pred_(3, i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        if ( fabs(p_x) > min_val || fabs(p_y) > min_val ) {
            // measurement model
            Zsig(0, i) = sqrt(p_x * p_x + p_y * p_y);                             //r
            Zsig(1, i) = atan2(p_y, p_x);                                     //phi
            Zsig(2, i) = (p_x * v1 + p_y * v2) / sqrt(p_x * p_x + p_y * p_y); //r_dot
        } else {
            Zsig(0, i) = 0;
            Zsig(1, i) = 0;
            Zsig(2, i) = 0;
        }
    }

    //mean predicted measurement
    VectorXd z_pred = VectorXd::Zero(n_z_radar_);
    for (int i=0; i < n_sig_; i++) {
        z_pred += weights_(i) * Zsig.col(i);
    }

    //measurement covariance matrix S
    MatrixXd S = MatrixXd::Zero(n_z_radar_, n_z_radar_);
    for (int i = 0; i < n_sig_; i++) {  //n_sig_ simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        Tools::NormalizeAngle(z_diff(1));

        S = S + weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    S = S + R_radar_;

    //
    // 2. Update the state and the state covariance matrix
    //
    // set measurement vector
    VectorXd z = VectorXd(n_z_radar_);
    for (unsigned int i = 0; i < n_z_radar_; i++)
        z(i) = meas_package.raw_measurements_[i];

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd::Zero(n_x_, n_z_radar_);
    for (int i = 0; i < n_sig_; i++) {  //n_sig_ simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        Tools::NormalizeAngle(z_diff(1));
        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        Tools::NormalizeAngle(x_diff(3));

        Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    Tools::NormalizeAngle(z_diff(1));

    //update state mean and covariance matrix
    double NIS = UpdateState(K, z_diff, S);
    cout << "NIS(radar) = " << NIS << endl;
}

/**
 * Updates the state and the state covariance matrix
 * @param {MatrixXd} K Kalman gain
 * @param {MatrixXd} z_diff measurement difference
 * @param {MatrixXd} S measurement covariance
 */
double UKF::UpdateState(const MatrixXd &K, const MatrixXd &z_diff, const MatrixXd &S)
{
    //update state mean and covariance matrix
    x_ = x_ + K * z_diff;
    P_ = P_ - K * S * K.transpose();

    // return NIS
    return (z_diff.transpose() * S.inverse() * z_diff)(0, 0);
}
