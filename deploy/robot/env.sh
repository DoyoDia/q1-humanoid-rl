# ROS 2 environment of the robot's own services (read from mc_app_main's /proc/<pid>/environ).
source /opt/ros/humble/setup.bash
# aimdk_msgs: app_proxy/setup.bash on the old firmware, interface/setup.bash on SDK v4 robots
for f in /robot/software/app_proxy/setup.bash /robot/software/interface/setup.bash; do
  [ -f "$f" ] && { source "$f"; break; }
done
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0
export FASTRTPS_DEFAULT_PROFILES_FILE=/robot/software/entry/cfg/standard/ros_dds_configuration.xml
