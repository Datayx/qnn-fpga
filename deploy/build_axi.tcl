# Vitis HLS: C-sim, synthesise and export the qnn_axi wrapper as a Vivado IP.
# Env: QNN_HLS_DIR = hls4ml project (default ../hls/mlp_4b relative to deploy/), run from deploy/build/.
# Matches the hls4ml project settings (part, 5 ns clock, uncertainty, compile/schedule options).
set here [file normalize [file dirname [info script]]]
if {[info exists ::env(QNN_HLS_DIR)]} { set hls_dir [file normalize $::env(QNN_HLS_DIR)] } \
else { set hls_dir [file normalize [file join $here .. hls mlp_4b]] }
source [file join $hls_dir project.tcl]
set fw [file join $hls_dir firmware]
set cflags "-std=c++0x -I$fw -I$here"

open_project -reset qnn_axi_prj
set_top qnn_axi
add_files [file join $here qnn_axi.cpp] -cflags $cflags
add_files [file join $fw myproject.cpp] -cflags $cflags
add_files -tb [file join $here qnn_axi_tb.cpp] -cflags $cflags
add_files -tb [file join $fw weights]   ;# C-sim loads weights/*.txt at run time, as in hls4ml's build_prj.tcl

open_solution -reset solution1
set_part $part
config_compile -name_max_length 80
config_schedule -enable_dsp_full_reg=false
create_clock -period $clock_period -name default
set_clock_uncertainty $clock_uncertainty default

puts "***** qnn_axi: hls_dir=$hls_dir part=$part clk=$clock_period unc=$clock_uncertainty *****"
csim_design
csynth_design
export_design -format ip_catalog -vendor qnn -library hls -ipname qnn_axi -version 1.0.0
exit
