module delay (
    (* iopad_external_pin, clkbuf_inhibit *) input clk,
    (* iopad_external_pin *) input in_clk,
    (* iopad_external_pin *) input cs,
    (* iopad_external_pin *) input sdo,
    (* iopad_external_pin *) output miso,
    (* iopad_external_pin *) output miso_en,
    (* iopad_external_pin *) output o_clk,
    (* iopad_external_pin *) output o_clk_en,
    (* iopad_external_pin *) output o_cs,
    (* iopad_external_pin *) output o_cs_en,
    (* iopad_external_pin *) output clk_en,
    (* iopad_external_pin *) input nreset
);
    reg [6:0] shift_reg = 7'b1111111;
    reg o_cs_reg;
    assign o_cs = o_cs_reg;
    assign miso_en = 1'b1;
    assign o_cs_en = 1'b1;
    assign o_clk_en = 1'b1;
    assign clk_en = 1'b1;
    reg sdo_reg;
    assign miso = sdo_reg;
    reg o_clk_reg;
    assign o_clk = o_clk_reg;
    reg in_clk_sync1;
    reg in_clk_sync2;
    reg cs_sync;
    reg nrst;
    always @(posedge clk) begin
        nrst <= nreset;
    end
    always @(posedge clk) begin
        in_clk_sync1 <= in_clk;
        in_clk_sync2 <= in_clk_sync1;
        cs_sync <= cs;
        if (in_clk_sync1 && ~in_clk_sync2) begin
            shift_reg = {shift_reg[5:0], cs_sync};
        end
        o_clk_reg = in_clk_sync1;
        o_cs_reg = shift_reg[6];
        if (cs) begin
            shift_reg = 7'b1111111;
        end
    end
    always @(posedge clk) begin
        sdo_reg <= sdo;
    end
endmodule
