// Starts qnn_axi and measures its on-chip latency in clock cycles.
//   go      : level from AXI GPIO ch1 bit 0; a rising edge starts one inference
//   status  : to AXI GPIO ch2 = {done_flag, count[30:0]}
// count = clock cycles from the first cycle ap_start is high up to and including the cycle ap_done
// is high. For a core with HLS latency L this reads L + 1 (the start cycle itself is counted).
// Inputs/outputs of the core are written/read over its own AXI-Lite port before/after.
module latency_ctrl (
    input  wire        clk,
    input  wire        rstn,
    input  wire        go,
    output reg         ap_start,
    input  wire        ap_ready,
    input  wire        ap_done,
    input  wire        ap_idle,
    output wire [31:0] status
);
    reg        go_d;
    reg        busy;
    reg        done_flag;
    reg [30:0] count;

    always @(posedge clk) begin
        if (!rstn) begin
            go_d <= 1'b0; ap_start <= 1'b0; busy <= 1'b0; done_flag <= 1'b0; count <= 31'd0;
        end else begin
            go_d <= go;
            if (go && !go_d && !busy) begin
                ap_start <= 1'b1; busy <= 1'b1; done_flag <= 1'b0; count <= 31'd0;
            end else begin
                if (ap_start && ap_ready) ap_start <= 1'b0;
                if (busy) begin
                    count <= count + 31'd1;
                    if (ap_done) begin busy <= 1'b0; done_flag <= 1'b1; end
                end
            end
        end
    end

    assign status = {done_flag, count};
endmodule
