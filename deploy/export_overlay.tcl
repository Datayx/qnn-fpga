# Export the PYNQ overlay (qnn.bit + qnn.hwh), reports and address map from a finished impl_1.
# Sourced at the end of kr260_bd.tcl, or standalone on an existing project:
#   vivado -mode batch -source deploy/export_overlay.tcl   (QNN_BUILD as in kr260_bd.tcl)
if {![info exists here]} { set here [file normalize [file dirname [info script]]] }
if {![info exists bdir]} {
    set bdir [expr {[info exists ::env(QNN_BUILD)] ? $::env(QNN_BUILD) : "$::env(HOME)/qnn_build/vivado"}]
}
set outdir [file join $here overlay]
if {[llength [get_projects -quiet]] == 0} { open_project [file join $bdir kr260_qnn.xpr] }
file mkdir $outdir

open_run impl_1
report_timing_summary -max_paths 5 -file [file join $outdir timing.rpt]
report_utilization -hierarchical -file [file join $outdir utilization_hier.rpt]
report_utilization -file [file join $outdir utilization.rpt]
set wns [get_property SLACK [lindex [get_timing_paths -setup -max_paths 1] 0]]
set clk_ns {}
foreach c [get_clocks] { lappend clk_ns "[get_property NAME $c]=[get_property PERIOD $c]" }

file copy -force [file join $bdir kr260_qnn.runs impl_1 qnn_wrapper.bit] [file join $outdir qnn.bit]
file copy -force [file join $bdir kr260_qnn.gen sources_1 bd qnn hw_handoff qnn.hwh] [file join $outdir qnn.hwh]

open_bd_design [get_files qnn.bd]
set fh [open [file join $outdir address_map.txt] w]
foreach s [get_bd_addr_segs -of_objects [get_bd_addr_spaces zynq_ps/Data]] {
    puts $fh "[get_property NAME $s] [get_property OFFSET $s] [get_property RANGE $s]"
}
puts $fh "core_clk_hz [get_property CONFIG.FREQ_HZ [get_bd_pins clk_wiz/clk_out1]]"
puts $fh "clocks_ns $clk_ns"
puts $fh "wns_ns $wns"
close $fh
puts "QNN_DONE WNS=$wns -> $outdir/qnn.bit, qnn.hwh"
