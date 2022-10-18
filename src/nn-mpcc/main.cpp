// Copyright 2019 Alexander Liniger

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////
#include <ros/ros.h>
#include <string>
#include <sstream>
#include <std_msgs/Float32MultiArray.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>

#include "Tests/spline_test.h"
#include "Tests/model_integrator_test.h"
#include "Tests/constratins_test.h"
#include "Tests/cost_test.h"
#include "MPC/mpc.h"
#include "Model/integrator.h"
#include "Params/track.h"
#include "Plotting/plotting.h"  

#include <nlohmann/json.hpp> 

using json = nlohmann::json;
using namespace mpcc;

State x0 = {0,0,0,0,0,0,0.25,0,0,0};   // D=0.25
double Ts = 0.02;
Shift shift = {0, 0, 0};

double quad2euler(double x, double y, double z, double w){

    return atan2(2 * (x*y+w*z), w*w+x*x-y*y-z*z);
}

void odomCallback(nav_msgs::OdometryPtr msg){
    x0.X = msg->pose.pose.position.x - shift.x;
    x0.Y = msg->pose.pose.position.y - shift.x;
    
    double x = msg->pose.pose.orientation.x;
    double y = msg->pose.pose.orientation.y;
    double z = msg->pose.pose.orientation.z;
    double w = msg->pose.pose.orientation.w;
    x0.phi = quad2euler(x, y, z, w) - shift.phi; 

    x0.vx = std::max(0.05, msg->twist.twist.linear.x*cos(x0.phi)+ msg->twist.twist.linear.y*sin(x0.phi));
    x0.vy = -msg->twist.twist.linear.y*sin(x0.phi) + msg->twist.twist.linear.y*cos(x0.phi);
    x0.r = msg->twist.twist.linear.z;

    // x0.s = x0.s + x0.vs*Ts*0.5;
}

void remotecontrolCallback(const std_msgs::Float32MultiArray &msg){
    // x0.D = msg.data[0];
    // x0.delta = msg.data[1];
} 

int main(int argc, char** argv) {
    
    ros::init(argc, argv, "mpc");
    ros::NodeHandle n("~");
    ros::Subscriber odom_sub = n.subscribe("/ekf/ekf_odom", 1, odomCallback, ros::TransportHints().tcpNoDelay());
    ros::Subscriber remotecontrol_sub = n.subscribe("/remote_control", 1, remotecontrolCallback, ros::TransportHints().tcpNoDelay());
    ros::Publisher control_pub = n.advertise<std_msgs::Float32MultiArray>("/computer_control", 10);
    ros::Publisher state_pub = n.advertise<std_msgs::Float32MultiArray>("/car_state", 10);
    ros::Rate loop_rate(10);    // 10 Hz 
    
    std::cout << "\n*********************************************" << std::endl;
    std::string path;
    n.getParam("config_path", path);
    std::ifstream iConfig(path);
    json jsonConfig;
    iConfig >> jsonConfig;
    
    PathToJson json_paths {jsonConfig["model_path"],
                           jsonConfig["cost_path"],
                           jsonConfig["bounds_path"],
                           jsonConfig["track_path"],
                           jsonConfig["normalization_path"]};
    int n_sqp, n_reset, run_time;
    double sqp_mixing, Ts;
    n.getParam("Ts", Ts);   
    n.getParam("n_sqp", n_sqp);
    n.getParam("n_reset", n_reset);
    n.getParam("run_time", run_time);
    n.getParam("sqp_mixing", sqp_mixing);
    n.getParam("x_shift", shift.x);
    n.getParam("y_shift", shift.y);
    n.getParam("phi_shift", shift.phi);

    // Integrator integrator = Integrator(Ts,json_paths);
    Plotting plotter = Plotting(Ts,json_paths);
    Track track = Track(json_paths.track_path); 
    TrackPos track_xy = track.getTrack();

    // the initial value of state(probably dosen't matter)
    x0.X = track_xy.X(0);
    x0.Y = track_xy.Y(0);
    x0.phi = std::atan2(track_xy.Y(1) - track_xy.Y(0),track_xy.X(1) - track_xy.X(0));
    n.getParam("v0", x0.vx);
    n.getParam("v0", x0.vs);
    
    // set MPC 
    MPC mpc(n_sqp, n_reset, sqp_mixing, Ts, json_paths);
    mpc.setTrack(track_xy.X,track_xy.Y);
    std::list<MPCReturn> log;

    // starting 
    n.getParam("D0", x0.D);
    std_msgs::Float32MultiArray control_msg;
    control_msg.data.clear();
    control_msg.data.push_back(x0.D);
    control_msg.data.push_back(x0.delta);

    int i = 0;
    while (i < 6)
    {   
        ros::spinOnce();
        std_msgs::Float32MultiArray state_msg;
        state_msg.data.clear();
        state_msg.data.push_back(x0.X);
        state_msg.data.push_back(x0.Y);
        state_msg.data.push_back(x0.phi);
        state_msg.data.push_back(x0.vx);
        state_msg.data.push_back(x0.vy);
        state_msg.data.push_back(x0.r);
        state_msg.data.push_back(x0.s);
        state_msg.data.push_back(x0.vs);
        state_pub.publish(state_msg);

        control_pub.publish(control_msg);

        i++;
        loop_rate.sleep();
    }
    
    // while (ros::ok() && i < run_time)
    while(ros::ok())
    {
        ros::spinOnce();
        // publish current state 
        std_msgs::Float32MultiArray state_msg;
        state_msg.data.clear();
        state_msg.data.push_back(x0.X);
        state_msg.data.push_back(x0.Y);
        state_msg.data.push_back(x0.phi);
        state_msg.data.push_back(x0.vx);
        state_msg.data.push_back(x0.vy);
        state_msg.data.push_back(x0.r);
        state_msg.data.push_back(x0.s);
        state_msg.data.push_back(x0.vs);
        state_pub.publish(state_msg);

        MPCReturn mpc_sol = mpc.runMPC(x0, i, 1);

        // state_msg.data.clear();
        // state_msg.data.push_back(x0.X);
        // state_msg.data.push_back(x0.Y);
        // state_msg.data.push_back(x0.phi);
        // state_msg.data.push_back(x0.vx);
        // state_msg.data.push_back(x0.vy);
        // state_msg.data.push_back(x0.r);
        // state_msg.data.push_back(x0.s);
        // state_msg.data.push_back(x0.vs);
        // state_pub.publish(state_msg);

        // update the control value
        x0.D = x0.D + mpc_sol.u0.dD;
        x0.delta = x0.delta + mpc_sol.u0.dDelta;
        x0.vs = x0.vs + mpc_sol.u0.dVs;
        // publish control value
        control_msg.data.clear();
        control_msg.data.push_back(x0.D);
        control_msg.data.push_back(x0.delta);
        control_pub.publish(control_msg);
        // iterate  
        log.push_back(mpc_sol);
        i++;
        loop_rate.sleep();
    }
  
    // plotter.plotRun(log,track_xy);
    // plotter.plotSim(log,track_xy);

    // double mean_time1 = 0.0, mean_time2 = 0.0;
    // double max_time1 = 0.0, max_time2 = 0.0, secmax_time2 = 0.0;
    // for(MPCReturn log_i : log)
    // {
    //     mean_time1 += log_i.time_total;
    //     if(log_i.time_total > max_time1)
    //         max_time1 = log_i.time_total;
    // }
    // std::cout << "mean nmpc time " << mean_time1/double(jsonConfig["n_sim"]);
    // // << "  " << mean_time2/double(jsonConfig["n_sim"]) << std::endl;
    // std::cout << "max nmpc time " << max_time1; 
    // // << "  " << max_time2  << " " << secmax_time2 << std::endl;
    return 0;
}


