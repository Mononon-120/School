(* top *) module delay_extclk (
  (* iopad_external_pin, clkbuf_inhibit *) input lac0_clk, //LOGIC_AS_CLK0
  (* iopad_external_pin, clkbuf_inhibit *) input lac1_clk, //LOGIC_AS_CLK1
  (* iopad_external_pin *) input nreset, //FPGA_CORE_READY
  (* iopad_external_pin *) input ext_clk_in,
  (* iopad_external_pin *) input sclk_in,
  (* iopad_external_pin *) input cs,
  (* iopad_external_pin *) input sdo,
  (* iopad_external_pin *) output ref_lac0_out, //REF_LOGIC_AS_CLK0
  (* iopad_external_pin *) output ref_lac1_out, //REF_LOGIC_AS_CLK1
  (* iopad_external_pin *) output lac0_en, //LOGIC_AS_CLK0_EN
  (* iopad_external_pin *) output lac1_en, //LOGIC_AS_CLK1_EN
  (* iopad_external_pin *) output div16_out,
  (* iopad_external_pin *) output div16_out_oe,
  (* iopad_external_pin *) output miso,
  (* iopad_external_pin *) output miso_en,
  (* iopad_external_pin *) output sclk_out,
  (* iopad_external_pin *) output sclk_out_en,
  (* iopad_external_pin *) output cs_out,
  (* iopad_external_pin *) output cs_out_en
);
  assign lac0_en = 1'b1;
  assign lac1_en = 1'b1;
  assign ref_lac0_out = ext_clk_in;
  reg nrst0;
  reg [3:0] div_cnt;
  wire div16_clk = div_cnt[3];
  always @(posedge lac0_clk) begin
    nrst0 <= nreset;
  end
  always @(posedge lac0_clk) begin
    if (!nrst0)
      div_cnt <= 4'b0000;
    else
      div_cnt <= div_cnt + 4'b0001;
  end
  assign ref_lac1_out = div16_clk;
  assign div16_out = div16_clk;
  assign div16_out_oe = 1'b1;
  assign miso_en = 1'b1;
  assign sclk_out_en = 1'b1;
  assign cs_out_en = 1'b1;
  reg [6:0] shift_reg = 7'b1111111;
  assign cs_out = shift_reg[6];
  assign miso = sdo;
  assign sclk_out = sclk_in;
  reg cs_sync;
  reg nrst1;
  reg event_ff1, event_ff2;
  reg event_prev;
  always @(posedge lac1_clk) begin
    nrst1 <= nreset;
  end
  always @(posedge lac1_clk) begin
    if (!nrst1) begin
      event_ff1 <= 1'b0;
      event_ff2 <= 1'b0;
      event_prev <= 1'b0;
      shift_reg <= 7'b1111111;
      cs_sync <= 1'b0;
    end else begin
      event_ff1 <= sclk_in;
      event_ff2 <= event_ff1;
      cs_sync <= cs;
      if (event_ff2 & ~event_prev)
        shift_reg <= {shift_reg[5:0], cs_sync};
      event_prev <= event_ff2;
      if (cs_sync)
        shift_reg <= 7'b1111111;
    end
  end
endmodule
