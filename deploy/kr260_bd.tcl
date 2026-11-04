# KR260 block design: Zynq MPSoC -> SmartConnect -> qnn_axi (AXI-Lite data) + AXI GPIO (latency_ctrl).
# Builds the bitstream and copies qnn.bit / qnn.hwh (PYNQ overlay) + reports to deploy/overlay/.
#
# Env: QNN_BUILD    project dir (default ~/qnn_build/vivado, WSL ext4 is much faster than /mnt/c)
#      QNN_FREQ_MHZ PL clock for the core and the AXI bus (default 200)
#      VIVADO_THREADS
# Run via: bash deploy/run_bd.sh
#
# Address map (assigned by Vivado, printed at the end and saved to overlay/address_map.txt):
#   qnn_axi_0/s_axi_ctrl : in0..in7 at 0x10.. (see the HLS driver header xqnn_axi_hw.h), out0..out2 + vld
#   axi_gpio_0           : ch1 (0x0) = go, ch2 (0x8) = {done, count[30:0]}
set here    [file normalize [file dirname [info script]]]
set part    xck26-sfvc784-2LV-c
set freq    [expr {[info exists ::env(QNN_FREQ_MHZ)] ? $::env(QNN_FREQ_MHZ) : 200}]
set bdir    [expr {[info exists ::env(QNN_BUILD)] ? $::env(QNN_BUILD) : "$::env(HOME)/qnn_build/vivado"}]
set ip_repo [file join $here build qnn_axi_prj solution1 impl ip]
set outdir  [file join $here overlay]
if {[info exists ::env(VIVADO_THREADS)]} { set_param general.maxThreads $::env(VIVADO_THREADS) }

create_project -force kr260_qnn $bdir -part $part
set bp [lindex [lsort [get_board_parts -quiet xilinx.com:kr260_som:*]] end]
puts "QNN: board_part=$bp freq=${freq} MHz build=$bdir"
set_property board_part $bp [current_project]
set_property ip_repo_paths $ip_repo [current_project]
update_ip_catalog
add_files -norecurse [file join $here latency_ctrl.v]

create_bd_design qnn

# --- PS: board preset, one 32-bit FPD master, one fabric clock + reset
set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e zynq_ps]
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1"} $ps
set_property -dict [list \
    CONFIG.PSU__USE__M_AXI_GP0 {1} CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} \
    CONFIG.PSU__USE__M_AXI_GP1 {0} CONFIG.PSU__USE__M_AXI_GP2 {0} \
    CONFIG.PSU__FPGA_PL0_ENABLE {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {100} \
    CONFIG.PSU__NUM_FABRIC_RESETS {1}] $ps

# --- 100 MHz pl_clk0 -> Clocking Wizard -> core/bus clock
set cw [create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wiz clk_wiz]
set_property -dict [list CONFIG.CLKOUT1_REQUESTED_OUT_FREQ $freq \
    CONFIG.RESET_TYPE {ACTIVE_LOW} CONFIG.RESET_PORT {resetn} CONFIG.USE_LOCKED {true}] $cw
connect_bd_net [get_bd_pins zynq_ps/pl_clk0] [get_bd_pins clk_wiz/clk_in1]
# pl_clk0 is really 99.999001 MHz; take the input frequency from the PS instead of hard-coding 100
set_property CONFIG.PRIM_IN_FREQ.VALUE_SRC PROPAGATED $cw
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins clk_wiz/resetn]

set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset rst_core]
connect_bd_net [get_bd_pins clk_wiz/clk_out1] [get_bd_pins rst_core/slowest_sync_clk]
connect_bd_net [get_bd_pins zynq_ps/pl_resetn0] [get_bd_pins rst_core/ext_reset_in]
connect_bd_net [get_bd_pins clk_wiz/locked] [get_bd_pins rst_core/dcm_locked]
set clk  [get_bd_pins clk_wiz/clk_out1]
set rstn [get_bd_pins rst_core/peripheral_aresetn]

# --- interconnect, core, GPIO, latency counter
set smc [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect smc]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {2}] $smc
set qnn [create_bd_cell -type ip -vlnv qnn:hls:qnn_axi qnn_axi_0]
set gpio [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_gpio axi_gpio_0]
set_property -dict [list CONFIG.C_IS_DUAL {1} CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_GPIO_WIDTH {1} \
    CONFIG.C_ALL_INPUTS_2 {1} CONFIG.C_GPIO2_WIDTH {32}] $gpio
set lc [create_bd_cell -type module -reference latency_ctrl latency_ctrl_0]

connect_bd_intf_net [get_bd_intf_pins zynq_ps/M_AXI_HPM0_FPD] [get_bd_intf_pins smc/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc/M00_AXI] [get_bd_intf_pins qnn_axi_0/s_axi_ctrl]
connect_bd_intf_net [get_bd_intf_pins smc/M01_AXI] [get_bd_intf_pins axi_gpio_0/S_AXI]

connect_bd_net $clk [get_bd_pins zynq_ps/maxihpm0_fpd_aclk] [get_bd_pins smc/aclk] \
    [get_bd_pins qnn_axi_0/ap_clk] [get_bd_pins axi_gpio_0/s_axi_aclk] [get_bd_pins latency_ctrl_0/clk]
connect_bd_net [get_bd_pins rst_core/interconnect_aresetn] [get_bd_pins smc/aresetn]
connect_bd_net $rstn [get_bd_pins qnn_axi_0/ap_rst_n] [get_bd_pins axi_gpio_0/s_axi_aresetn] \
    [get_bd_pins latency_ctrl_0/rstn]

connect_bd_net [get_bd_pins axi_gpio_0/gpio_io_o] [get_bd_pins latency_ctrl_0/go]
connect_bd_net [get_bd_pins latency_ctrl_0/status] [get_bd_pins axi_gpio_0/gpio2_io_i]
foreach p {ap_start} { connect_bd_net [get_bd_pins latency_ctrl_0/$p] [get_bd_pins qnn_axi_0/$p] }
foreach p {ap_ready ap_done ap_idle} { connect_bd_net [get_bd_pins qnn_axi_0/$p] [get_bd_pins latency_ctrl_0/$p] }

assign_bd_address
validate_bd_design
save_bd_design

# --- synth / impl / bitstream
set bd_file [get_files qnn.bd]
make_wrapper -files $bd_file -top
add_files -norecurse [file join $bdir kr260_qnn.gen sources_1 bd qnn hdl qnn_wrapper.v]
set_property top qnn_wrapper [current_fileset]
generate_target all $bd_file
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} { error "QNN: impl_1 did not complete" }

# --- outputs (separate script so it can be rerun on a finished project)
source [file join $here export_overlay.tcl]
