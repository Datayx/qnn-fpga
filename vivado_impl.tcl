# Out-of-context synthesis + place & route of an hls4ml IP, with a real clock constraint.
# Gives post-route utilisation and timing (WNS), which HLS C-synthesis only estimates.
#
# usage: vivado -mode batch -nojournal -nolog -source vivado_impl.tcl -tclargs <hls_dir> <part> <period_ns>
# writes <hls_dir>/impl/{post_route_util.rpt, post_route_timing.rpt, summary.txt}
#
# Note: out-of-context means no I/O buffers and unconstrained input/output delays, so only
# register-to-register paths are timed. That is the standard way to quote an IP block's Fmax.

lassign $argv hls_dir part period
# Fewer threads = lower peak memory; set VIVADO_THREADS=1 or 2 for the biggest designs on a 16 GB laptop.
if {[info exists ::env(VIVADO_THREADS)]} { set_param general.maxThreads $::env(VIVADO_THREADS) }
set top myproject
set vdir [file join $hls_dir ${top}_prj solution1 syn verilog]
set odir [file join $hls_dir impl]
file mkdir $odir

read_verilog [glob [file join $vdir *.v]]
set xdc [file join $odir clock.xdc]
set fh [open $xdc w]
puts $fh "create_clock -period $period -name ap_clk \[get_ports ap_clk\]"
close $fh
read_xdc -mode out_of_context $xdc

synth_design -top $top -part $part -mode out_of_context -flatten_hierarchy rebuilt
report_utilization -file [file join $odir post_synth_util.rpt]
opt_design
place_design
phys_opt_design
route_design

report_utilization -file [file join $odir post_route_util.rpt]
report_timing_summary -max_paths 5 -file [file join $odir post_route_timing.rpt]

report_route_status -file [file join $odir route_status.rpt]

set path [lindex [get_timing_paths -setup -max_paths 1] 0]
set wns [get_property SLACK $path]
set bad [llength [get_nets -hier -quiet -filter {ROUTE_STATUS == UNROUTED || ROUTE_STATUS == PARTIAL || ROUTE_STATUS == CONFLICTS}]]
set fh [open [file join $odir summary.txt] w]
puts $fh "WNS $wns"
puts $fh "PERIOD $period"
puts $fh "BAD_NETS $bad"
close $fh
