`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: NTU Singapore
// Engineer: Arish S
//
// Create Date: 2019-10-29 22:50:41
// Design Name: 
// Module Name: top
// Project Name: 
// Target Devices:
// Tool Versions: 
// Description: 
//
// Dependencies: 
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
//
//////////////////////////////////////////////////////////////////////////////////


module top#(
  parameter integer BITWIDTH_IN=8,
  parameter integer BITWIDTH_W=8,
  parameter integer BITWIDTH_B=8,
  parameter integer BITWIDTH_C1=16,
  parameter integer KERNELHEIGHT_C1=3,
  parameter integer KERNELWIDTH_C1=3,
  parameter integer DATAWIDTH_IN=28,
  parameter integer DATAHEIGHT_IN=28,
  parameter integer STRIDEWIDTH_C1=1,
  parameter integer STRIDEHEIGHT_C1=1,
  parameter integer FILTERBATCH_C1=6,
  parameter integer DATAWIDTH_M1=26,
  parameter integer DATAHEIGHT_M1=26,
  parameter integer BITWIDTH_M1=16,
  parameter integer STRIDEWIDTH_POOL_M1=2,
  parameter integer STRIDEHEIGHT_POOL_M1=2,
  parameter integer KERNELWIDTH_POOL_M1=2,
  parameter integer KERNELHEIGHT_POOL_M1=2,
  parameter integer DATAWIDTH_C2=13,
  parameter integer DATAHEIGHT_C2=13,
  parameter integer BITWIDTH_C2=16,
  parameter integer KERNELHEIGHT_C2=3,
  parameter integer KERNELWIDTH_C2=3,
  parameter integer STRIDEWIDTH_C2=1,
  parameter integer STRIDEHEIGHT_C2=1,
  parameter integer FILTERBATCH_C2=16,
  parameter integer DATAWIDTH_M2=11,
  parameter integer DATAHEIGHT_M2=11,
  parameter integer BITWIDTH_M2=16,
  parameter integer STRIDEWIDTH_POOL_M2=2,
  parameter integer STRIDEHEIGHT_POOL_M2=2,
  parameter integer KERNELWIDTH_POOL_M2=3,
  parameter integer KERNELHEIGHT_POOL_M2=3,
  parameter integer DATAWIDTH_D1=5,
  parameter integer DATAHEIGHT_D1=5,
  parameter integer NEURONS_D1=10,
  parameter integer BITWIDTH_D1=32,
  parameter integer BITWIDTH_SM=32,
  parameter integer ADDR_WIDTH=11,
  parameter integer DATA_WIDTH=8,
  parameter integer DEPTH=2048,
  parameter integer OFFSET_C2_W=844,
  parameter integer OFFSET_C2_B=1708,
  parameter integer OFFSET_D1_W=1724,
  parameter integer OFFSET_D1_B=5724,
  parameter integer OFFSET_C1_W=784,
  parameter integer OFFSET_C1_B=838,
  parameter integer ADDR_WIDTH_BUF=9,
  parameter integer DATA_WIDTH_BUF=16,
  parameter integer DEPTH_BUF=512,
  parameter integer ADDR_WIDTH_BUF_C1=10,
  parameter integer DEPTH_BUF_C1=1024,
  parameter integer ADDR_WIDTH_BUF_M1=8,
  parameter integer DEPTH_BUF_M1=256,
  parameter integer ADDR_WIDTH_BUF_C2=7,
  parameter integer DEPTH_BUF_C2=128,
  parameter integer ADDR_WIDTH_BUF_M2=5,
  parameter integer DEPTH_BUF_M2=32
)
(
  input clk, in_ready,
  output reg [0:3] prediction,
  output reg done
  );
  
  reg done_c1, done_m1, done_c2, done_m2, done_den1, done_soft;
  reg rst_c1,rst_m1,rst_c2,rst_m2,rst_d1;
  reg en_c1,en_m1,en_c2,en_m2,en_d1,en_sm;
  reg [4:0] state=0;
  
  //**********************Dense D1 Layer*******************************//
  reg signed [0:319]fc_out;
  reg [1:0]count_c1=0;
  reg done_shift_c1;
  reg en_shift_c1,rst_shift_c1;
  reg [2:0] state_c1=0;
  ////////////////shift///////////////////////
  reg [10:0] addr, addr_c2, addr_d1;
  reg wr_en=0,wr_en_c2=0, wr_en_d1=0;
  reg oe=1, oe_c2=1;
  reg [7:0] din=0, din_c2=0, din_d1=0;
  wire [7:0] dout, dout_c2,dout_d1;
  reg done_load_shift_c1=0;
  reg [10:0] count_ld_shift_c1=0;
  reg signed [0:7] bias_shift_c1_ch1;
  reg signed [0:7] bias_shift_c1_ch2;
  reg signed [0:7] bias_shift_c1_ch3;
  reg signed [0:7] bias_shift_c1_ch4;
  reg signed [0:7] bias_shift_c1_ch5;
  reg signed [0:7] bias_shift_c1_ch6;
  reg signed [0:71] W_shift_c1_ch1;
  reg signed [0:71] W_shift_c1_ch2;
  reg signed [0:71] W_shift_c1_ch3;
  reg signed [0:71] W_shift_c1_ch4;
  reg signed [0:71] W_shift_c1_ch5;
  reg signed [0:71] W_shift_c1_ch6;
  reg [9:0]count_shift_c1=0;
  reg done_conv_c1;
  reg en_conv_c1;
  reg rst_conv_c1;
  reg signed [0:15] result_temp_conv_c1_ch1;
  reg signed [0:15] result_temp_conv_c1_ch2;
  reg signed [0:15] result_temp_conv_c1_ch3;
  reg signed [0:15] result_temp_conv_c1_ch4;
  reg signed [0:15] result_temp_conv_c1_ch5;
  reg signed [0:15] result_temp_conv_c1_ch6;
  reg [2:0] state_shift_c1=0;
  reg [9:0] window_select_c1;
  ///////////////////conv kernel////////////////////////
  reg done_load_conv_c1=0;
  reg [10:0] count_ld_conv_c1=0;
  wire [10:0] start_addr_conv_c1;
  reg signed [0:71] X_conv_c1;
  reg [3:0]count_conv_c1=0;
  reg en_m_c1;
  reg rst_m_c1;
  wire signed [7:0] mem_x_conv_c1 [0:8];
  wire done_m_c1_ch1;
  wire done_m_c1_ch2;
  wire done_m_c1_ch3;
  wire done_m_c1_ch4;
  wire done_m_c1_ch5;
  wire done_m_c1_ch6;
  wire signed [15:0] result_temp_m_c1_ch1;
  wire signed [15:0] result_temp_m_c1_ch2;
  wire signed [15:0] result_temp_m_c1_ch3;
  wire signed [15:0] result_temp_m_c1_ch4;
  wire signed [15:0] result_temp_m_c1_ch5;
  wire signed [15:0] result_temp_m_c1_ch6;
  reg signed [18:0] result_final_temp_conv_c1_ch1;
  reg signed [18:0] result_final_temp_conv_c1_ch2;
  reg signed [18:0] result_final_temp_conv_c1_ch3;
  reg signed [18:0] result_final_temp_conv_c1_ch4;
  reg signed [18:0] result_final_temp_conv_c1_ch5;
  reg signed [18:0] result_final_temp_conv_c1_ch6;
  reg signed [18:0] buffer_conv_c1_ch1=0;
  reg signed [18:0] buffer_conv_c1_ch2=0;
  reg signed [18:0] buffer_conv_c1_ch3=0;
  reg signed [18:0] buffer_conv_c1_ch4=0;
  reg signed [18:0] buffer_conv_c1_ch5=0;
  reg signed [18:0] buffer_conv_c1_ch6=0;
  wire signed [7:0] mem_w_conv_c1_ch1 [0:8];
  wire signed [7:0] mem_w_conv_c1_ch2 [0:8];
  wire signed [7:0] mem_w_conv_c1_ch3 [0:8];
  wire signed [7:0] mem_w_conv_c1_ch4 [0:8];
  wire signed [7:0] mem_w_conv_c1_ch5 [0:8];
  wire signed [7:0] mem_w_conv_c1_ch6 [0:8];
  reg signed [7:0] x_in_m_c1;
  reg signed [7:0] w_in_m_c1_ch1;
  reg signed [7:0] w_in_m_c1_ch2;
  reg signed [7:0] w_in_m_c1_ch3;
  reg signed [7:0] w_in_m_c1_ch4;
  reg signed [7:0] w_in_m_c1_ch5;
  reg signed [7:0] w_in_m_c1_ch6;
  reg [2:0] state_conv_c1=0;
  reg signed [0:15] THRESHOLD=0;
  
  reg [9:0] addr_buf_0_c1_ch1;
  reg wr_en_buf_0_c1_ch1=0, wr_en_buf_1_c1_ch1=0, oe_buf_0_c1_ch1=0, oe_buf_1_c1_ch1=0;
  reg [15:0] din_buf_0_c1_ch1;
  wire [15:0] dout_buf_0_c1_ch1;
  reg [9:0] addr_buf_1_c1_ch1;
  reg [15:0] din_buf_1_c1_ch1=0;
  wire [15:0] dout_buf_1_c1_ch1;
  
  reg [9:0] addr_buf_0_c1_ch2;
  reg wr_en_buf_0_c1_ch2=0, wr_en_buf_1_c1_ch2=0, oe_buf_0_c1_ch2=0, oe_buf_1_c1_ch2=0;
  reg [15:0] din_buf_0_c1_ch2;
  wire [15:0] dout_buf_0_c1_ch2;
  reg [9:0] addr_buf_1_c1_ch2;
  reg [15:0] din_buf_1_c1_ch2=0;
  wire [15:0] dout_buf_1_c1_ch2;
  
  reg [9:0] addr_buf_0_c1_ch3;
  reg wr_en_buf_0_c1_ch3=0, wr_en_buf_1_c1_ch3=0, oe_buf_0_c1_ch3=0, oe_buf_1_c1_ch3=0;
  reg [15:0] din_buf_0_c1_ch3;
  wire [15:0] dout_buf_0_c1_ch3;
  reg [9:0] addr_buf_1_c1_ch3;
  reg [15:0] din_buf_1_c1_ch3=0;
  wire [15:0] dout_buf_1_c1_ch3;
  
  reg [9:0] addr_buf_0_c1_ch4;
  reg wr_en_buf_0_c1_ch4=0, wr_en_buf_1_c1_ch4=0, oe_buf_0_c1_ch4=0, oe_buf_1_c1_ch4=0;
  reg [15:0] din_buf_0_c1_ch4;
  wire [15:0] dout_buf_0_c1_ch4;
  reg [9:0] addr_buf_1_c1_ch4;
  reg [15:0] din_buf_1_c1_ch4=0;
  wire [15:0] dout_buf_1_c1_ch4;
  
  reg [9:0] addr_buf_0_c1_ch5;
  reg wr_en_buf_0_c1_ch5=0, wr_en_buf_1_c1_ch5=0, oe_buf_0_c1_ch5=0, oe_buf_1_c1_ch5=0;
  reg [15:0] din_buf_0_c1_ch5;
  wire [15:0] dout_buf_0_c1_ch5;
  reg [9:0] addr_buf_1_c1_ch5;
  reg [15:0] din_buf_1_c1_ch5=0;
  wire [15:0] dout_buf_1_c1_ch5;
  
  reg [9:0] addr_buf_0_c1_ch6;
  reg wr_en_buf_0_c1_ch6=0, wr_en_buf_1_c1_ch6=0, oe_buf_0_c1_ch6=0, oe_buf_1_c1_ch6=0;
  reg [15:0] din_buf_0_c1_ch6;
  wire [15:0] dout_buf_0_c1_ch6;
  reg [9:0] addr_buf_1_c1_ch6;
  reg [15:0] din_buf_1_c1_ch6=0;
  wire [15:0] dout_buf_1_c1_ch6;
  
  reg [7:0] addr_buf_0_m1_ch1;
  reg wr_en_buf_0_m1_ch1=0, wr_en_buf_1_m1_ch1=0, oe_buf_0_m1_ch1=0, oe_buf_1_m1_ch1=0;
  reg [15:0] din_buf_0_m1_ch1;
  wire [15:0] dout_buf_0_m1_ch1;
  reg [7:0] addr_buf_1_m1_ch1;
  reg [15:0] din_buf_1_m1_ch1=0;
  wire [15:0] dout_buf_1_m1_ch1;
  
  reg [7:0] addr_buf_0_m1_ch2;
  reg wr_en_buf_0_m1_ch2=0, wr_en_buf_1_m1_ch2=0, oe_buf_0_m1_ch2=0, oe_buf_1_m1_ch2=0;
  reg [15:0] din_buf_0_m1_ch2;
  wire [15:0] dout_buf_0_m1_ch2;
  reg [7:0] addr_buf_1_m1_ch2;
  reg [15:0] din_buf_1_m1_ch2=0;
  wire [15:0] dout_buf_1_m1_ch2;
  
  reg [7:0] addr_buf_0_m1_ch3;
  reg wr_en_buf_0_m1_ch3=0, wr_en_buf_1_m1_ch3=0, oe_buf_0_m1_ch3=0, oe_buf_1_m1_ch3=0;
  reg [15:0] din_buf_0_m1_ch3;
  wire [15:0] dout_buf_0_m1_ch3;
  reg [7:0] addr_buf_1_m1_ch3;
  reg [15:0] din_buf_1_m1_ch3=0;
  wire [15:0] dout_buf_1_m1_ch3;
  
  reg [7:0] addr_buf_0_m1_ch4;
  reg wr_en_buf_0_m1_ch4=0, wr_en_buf_1_m1_ch4=0, oe_buf_0_m1_ch4=0, oe_buf_1_m1_ch4=0;
  reg [15:0] din_buf_0_m1_ch4;
  wire [15:0] dout_buf_0_m1_ch4;
  reg [7:0] addr_buf_1_m1_ch4;
  reg [15:0] din_buf_1_m1_ch4=0;
  wire [15:0] dout_buf_1_m1_ch4;
  
  reg [7:0] addr_buf_0_m1_ch5;
  reg wr_en_buf_0_m1_ch5=0, wr_en_buf_1_m1_ch5=0, oe_buf_0_m1_ch5=0, oe_buf_1_m1_ch5=0;
  reg [15:0] din_buf_0_m1_ch5;
  wire [15:0] dout_buf_0_m1_ch5;
  reg [7:0] addr_buf_1_m1_ch5;
  reg [15:0] din_buf_1_m1_ch5=0;
  wire [15:0] dout_buf_1_m1_ch5;
  
  reg [7:0] addr_buf_0_m1_ch6;
  reg wr_en_buf_0_m1_ch6=0, wr_en_buf_1_m1_ch6=0, oe_buf_0_m1_ch6=0, oe_buf_1_m1_ch6=0;
  reg [15:0] din_buf_0_m1_ch6;
  wire [15:0] dout_buf_0_m1_ch6;
  reg [7:0] addr_buf_1_m1_ch6;
  reg [15:0] din_buf_1_m1_ch6=0;
  wire [15:0] dout_buf_1_m1_ch6;
  
  reg [6:0] addr_buf_0_c2_ch1;
  reg wr_en_buf_0_c2_ch1=0, wr_en_buf_1_c2_ch1=0, oe_buf_0_c2_ch1=0, oe_buf_1_c2_ch1=0;
  reg [15:0] din_buf_0_c2_ch1;
  wire [15:0] dout_buf_0_c2_ch1;
  reg [6:0] addr_buf_1_c2_ch1;
  reg [15:0] din_buf_1_c2_ch1=0;
  wire [15:0] dout_buf_1_c2_ch1;
  
  reg [6:0] addr_buf_0_c2_ch2;
  reg wr_en_buf_0_c2_ch2=0, wr_en_buf_1_c2_ch2=0, oe_buf_0_c2_ch2=0, oe_buf_1_c2_ch2=0;
  reg [15:0] din_buf_0_c2_ch2;
  wire [15:0] dout_buf_0_c2_ch2;
  reg [6:0] addr_buf_1_c2_ch2;
  reg [15:0] din_buf_1_c2_ch2=0;
  wire [15:0] dout_buf_1_c2_ch2;
  
  reg [6:0] addr_buf_0_c2_ch3;
  reg wr_en_buf_0_c2_ch3=0, wr_en_buf_1_c2_ch3=0, oe_buf_0_c2_ch3=0, oe_buf_1_c2_ch3=0;
  reg [15:0] din_buf_0_c2_ch3;
  wire [15:0] dout_buf_0_c2_ch3;
  reg [6:0] addr_buf_1_c2_ch3;
  reg [15:0] din_buf_1_c2_ch3=0;
  wire [15:0] dout_buf_1_c2_ch3;
  
  reg [6:0] addr_buf_0_c2_ch4;
  reg wr_en_buf_0_c2_ch4=0, wr_en_buf_1_c2_ch4=0, oe_buf_0_c2_ch4=0, oe_buf_1_c2_ch4=0;
  reg [15:0] din_buf_0_c2_ch4;
  wire [15:0] dout_buf_0_c2_ch4;
  reg [6:0] addr_buf_1_c2_ch4;
  reg [15:0] din_buf_1_c2_ch4=0;
  wire [15:0] dout_buf_1_c2_ch4;
  
  reg [6:0] addr_buf_0_c2_ch5;
  reg wr_en_buf_0_c2_ch5=0, wr_en_buf_1_c2_ch5=0, oe_buf_0_c2_ch5=0, oe_buf_1_c2_ch5=0;
  reg [15:0] din_buf_0_c2_ch5;
  wire [15:0] dout_buf_0_c2_ch5;
  reg [6:0] addr_buf_1_c2_ch5;
  reg [15:0] din_buf_1_c2_ch5=0;
  wire [15:0] dout_buf_1_c2_ch5;
  
  reg [6:0] addr_buf_0_c2_ch6;
  reg wr_en_buf_0_c2_ch6=0, wr_en_buf_1_c2_ch6=0, oe_buf_0_c2_ch6=0, oe_buf_1_c2_ch6=0;
  reg [15:0] din_buf_0_c2_ch6;
  wire [15:0] dout_buf_0_c2_ch6;
  reg [6:0] addr_buf_1_c2_ch6;
  reg [15:0] din_buf_1_c2_ch6=0;
  wire [15:0] dout_buf_1_c2_ch6;
  
  reg [6:0] addr_buf_0_c2_ch7;
  reg wr_en_buf_0_c2_ch7=0, wr_en_buf_1_c2_ch7=0, oe_buf_0_c2_ch7=0, oe_buf_1_c2_ch7=0;
  reg [15:0] din_buf_0_c2_ch7;
  wire [15:0] dout_buf_0_c2_ch7;
  reg [6:0] addr_buf_1_c2_ch7;
  reg [15:0] din_buf_1_c2_ch7=0;
  wire [15:0] dout_buf_1_c2_ch7;
  
  reg [6:0] addr_buf_0_c2_ch8;
  reg wr_en_buf_0_c2_ch8=0, wr_en_buf_1_c2_ch8=0, oe_buf_0_c2_ch8=0, oe_buf_1_c2_ch8=0;
  reg [15:0] din_buf_0_c2_ch8;
  wire [15:0] dout_buf_0_c2_ch8;
  reg [6:0] addr_buf_1_c2_ch8;
  reg [15:0] din_buf_1_c2_ch8=0;
  wire [15:0] dout_buf_1_c2_ch8;
  
  reg [6:0] addr_buf_0_c2_ch9;
  reg wr_en_buf_0_c2_ch9=0, wr_en_buf_1_c2_ch9=0, oe_buf_0_c2_ch9=0, oe_buf_1_c2_ch9=0;
  reg [15:0] din_buf_0_c2_ch9;
  wire [15:0] dout_buf_0_c2_ch9;
  reg [6:0] addr_buf_1_c2_ch9;
  reg [15:0] din_buf_1_c2_ch9=0;
  wire [15:0] dout_buf_1_c2_ch9;
  
  reg [6:0] addr_buf_0_c2_ch10;
  reg wr_en_buf_0_c2_ch10=0, wr_en_buf_1_c2_ch10=0, oe_buf_0_c2_ch10=0, oe_buf_1_c2_ch10=0;
  reg [15:0] din_buf_0_c2_ch10;
  wire [15:0] dout_buf_0_c2_ch10;
  reg [6:0] addr_buf_1_c2_ch10;
  reg [15:0] din_buf_1_c2_ch10=0;
  wire [15:0] dout_buf_1_c2_ch10;
  
  reg [6:0] addr_buf_0_c2_ch11;
  reg wr_en_buf_0_c2_ch11=0, wr_en_buf_1_c2_ch11=0, oe_buf_0_c2_ch11=0, oe_buf_1_c2_ch11=0;
  reg [15:0] din_buf_0_c2_ch11;
  wire [15:0] dout_buf_0_c2_ch11;
  reg [6:0] addr_buf_1_c2_ch11;
  reg [15:0] din_buf_1_c2_ch11=0;
  wire [15:0] dout_buf_1_c2_ch11;
  
  reg [6:0] addr_buf_0_c2_ch12;
  reg wr_en_buf_0_c2_ch12=0, wr_en_buf_1_c2_ch12=0, oe_buf_0_c2_ch12=0, oe_buf_1_c2_ch12=0;
  reg [15:0] din_buf_0_c2_ch12;
  wire [15:0] dout_buf_0_c2_ch12;
  reg [6:0] addr_buf_1_c2_ch12;
  reg [15:0] din_buf_1_c2_ch12=0;
  wire [15:0] dout_buf_1_c2_ch12;
  
  reg [6:0] addr_buf_0_c2_ch13;
  reg wr_en_buf_0_c2_ch13=0, wr_en_buf_1_c2_ch13=0, oe_buf_0_c2_ch13=0, oe_buf_1_c2_ch13=0;
  reg [15:0] din_buf_0_c2_ch13;
  wire [15:0] dout_buf_0_c2_ch13;
  reg [6:0] addr_buf_1_c2_ch13;
  reg [15:0] din_buf_1_c2_ch13=0;
  wire [15:0] dout_buf_1_c2_ch13;
  
  reg [6:0] addr_buf_0_c2_ch14;
  reg wr_en_buf_0_c2_ch14=0, wr_en_buf_1_c2_ch14=0, oe_buf_0_c2_ch14=0, oe_buf_1_c2_ch14=0;
  reg [15:0] din_buf_0_c2_ch14;
  wire [15:0] dout_buf_0_c2_ch14;
  reg [6:0] addr_buf_1_c2_ch14;
  reg [15:0] din_buf_1_c2_ch14=0;
  wire [15:0] dout_buf_1_c2_ch14;
  
  reg [6:0] addr_buf_0_c2_ch15;
  reg wr_en_buf_0_c2_ch15=0, wr_en_buf_1_c2_ch15=0, oe_buf_0_c2_ch15=0, oe_buf_1_c2_ch15=0;
  reg [15:0] din_buf_0_c2_ch15;
  wire [15:0] dout_buf_0_c2_ch15;
  reg [6:0] addr_buf_1_c2_ch15;
  reg [15:0] din_buf_1_c2_ch15=0;
  wire [15:0] dout_buf_1_c2_ch15;
  
  reg [6:0] addr_buf_0_c2_ch16;
  reg wr_en_buf_0_c2_ch16=0, wr_en_buf_1_c2_ch16=0, oe_buf_0_c2_ch16=0, oe_buf_1_c2_ch16=0;
  reg [15:0] din_buf_0_c2_ch16;
  wire [15:0] dout_buf_0_c2_ch16;
  reg [6:0] addr_buf_1_c2_ch16;
  reg [15:0] din_buf_1_c2_ch16=0;
  wire [15:0] dout_buf_1_c2_ch16;
  
  reg [4:0] addr_buf_0_m2_ch1;
  reg wr_en_buf_0_m2_ch1=0, wr_en_buf_1_m2_ch1=0, oe_buf_0_m2_ch1=0, oe_buf_1_m2_ch1=0;
  reg [15:0] din_buf_0_m2_ch1;
  wire [15:0] dout_buf_0_m2_ch1;
  reg [4:0] addr_buf_1_m2_ch1;
  reg [15:0] din_buf_1_m2_ch1=0;
  wire [15:0] dout_buf_1_m2_ch1;
  
  reg [4:0] addr_buf_0_m2_ch2;
  reg wr_en_buf_0_m2_ch2=0, wr_en_buf_1_m2_ch2=0, oe_buf_0_m2_ch2=0, oe_buf_1_m2_ch2=0;
  reg [15:0] din_buf_0_m2_ch2;
  wire [15:0] dout_buf_0_m2_ch2;
  reg [4:0] addr_buf_1_m2_ch2;
  reg [15:0] din_buf_1_m2_ch2=0;
  wire [15:0] dout_buf_1_m2_ch2;
  
  reg [4:0] addr_buf_0_m2_ch3;
  reg wr_en_buf_0_m2_ch3=0, wr_en_buf_1_m2_ch3=0, oe_buf_0_m2_ch3=0, oe_buf_1_m2_ch3=0;
  reg [15:0] din_buf_0_m2_ch3;
  wire [15:0] dout_buf_0_m2_ch3;
  reg [4:0] addr_buf_1_m2_ch3;
  reg [15:0] din_buf_1_m2_ch3=0;
  wire [15:0] dout_buf_1_m2_ch3;
  
  reg [4:0] addr_buf_0_m2_ch4;
  reg wr_en_buf_0_m2_ch4=0, wr_en_buf_1_m2_ch4=0, oe_buf_0_m2_ch4=0, oe_buf_1_m2_ch4=0;
  reg [15:0] din_buf_0_m2_ch4;
  wire [15:0] dout_buf_0_m2_ch4;
  reg [4:0] addr_buf_1_m2_ch4;
  reg [15:0] din_buf_1_m2_ch4=0;
  wire [15:0] dout_buf_1_m2_ch4;
  
  reg [4:0] addr_buf_0_m2_ch5;
  reg wr_en_buf_0_m2_ch5=0, wr_en_buf_1_m2_ch5=0, oe_buf_0_m2_ch5=0, oe_buf_1_m2_ch5=0;
  reg [15:0] din_buf_0_m2_ch5;
  wire [15:0] dout_buf_0_m2_ch5;
  reg [4:0] addr_buf_1_m2_ch5;
  reg [15:0] din_buf_1_m2_ch5=0;
  wire [15:0] dout_buf_1_m2_ch5;
  
  reg [4:0] addr_buf_0_m2_ch6;
  reg wr_en_buf_0_m2_ch6=0, wr_en_buf_1_m2_ch6=0, oe_buf_0_m2_ch6=0, oe_buf_1_m2_ch6=0;
  reg [15:0] din_buf_0_m2_ch6;
  wire [15:0] dout_buf_0_m2_ch6;
  reg [4:0] addr_buf_1_m2_ch6;
  reg [15:0] din_buf_1_m2_ch6=0;
  wire [15:0] dout_buf_1_m2_ch6;
  
  reg [4:0] addr_buf_0_m2_ch7;
  reg wr_en_buf_0_m2_ch7=0, wr_en_buf_1_m2_ch7=0, oe_buf_0_m2_ch7=0, oe_buf_1_m2_ch7=0;
  reg [15:0] din_buf_0_m2_ch7;
  wire [15:0] dout_buf_0_m2_ch7;
  reg [4:0] addr_buf_1_m2_ch7;
  reg [15:0] din_buf_1_m2_ch7=0;
  wire [15:0] dout_buf_1_m2_ch7;
  
  reg [4:0] addr_buf_0_m2_ch8;
  reg wr_en_buf_0_m2_ch8=0, wr_en_buf_1_m2_ch8=0, oe_buf_0_m2_ch8=0, oe_buf_1_m2_ch8=0;
  reg [15:0] din_buf_0_m2_ch8;
  wire [15:0] dout_buf_0_m2_ch8;
  reg [4:0] addr_buf_1_m2_ch8;
  reg [15:0] din_buf_1_m2_ch8=0;
  wire [15:0] dout_buf_1_m2_ch8;
  
  reg [4:0] addr_buf_0_m2_ch9;
  reg wr_en_buf_0_m2_ch9=0, wr_en_buf_1_m2_ch9=0, oe_buf_0_m2_ch9=0, oe_buf_1_m2_ch9=0;
  reg [15:0] din_buf_0_m2_ch9;
  wire [15:0] dout_buf_0_m2_ch9;
  reg [4:0] addr_buf_1_m2_ch9;
  reg [15:0] din_buf_1_m2_ch9=0;
  wire [15:0] dout_buf_1_m2_ch9;
  
  reg [4:0] addr_buf_0_m2_ch10;
  reg wr_en_buf_0_m2_ch10=0, wr_en_buf_1_m2_ch10=0, oe_buf_0_m2_ch10=0, oe_buf_1_m2_ch10=0;
  reg [15:0] din_buf_0_m2_ch10;
  wire [15:0] dout_buf_0_m2_ch10;
  reg [4:0] addr_buf_1_m2_ch10;
  reg [15:0] din_buf_1_m2_ch10=0;
  wire [15:0] dout_buf_1_m2_ch10;
  
  reg [4:0] addr_buf_0_m2_ch11;
  reg wr_en_buf_0_m2_ch11=0, wr_en_buf_1_m2_ch11=0, oe_buf_0_m2_ch11=0, oe_buf_1_m2_ch11=0;
  reg [15:0] din_buf_0_m2_ch11;
  wire [15:0] dout_buf_0_m2_ch11;
  reg [4:0] addr_buf_1_m2_ch11;
  reg [15:0] din_buf_1_m2_ch11=0;
  wire [15:0] dout_buf_1_m2_ch11;
  
  reg [4:0] addr_buf_0_m2_ch12;
  reg wr_en_buf_0_m2_ch12=0, wr_en_buf_1_m2_ch12=0, oe_buf_0_m2_ch12=0, oe_buf_1_m2_ch12=0;
  reg [15:0] din_buf_0_m2_ch12;
  wire [15:0] dout_buf_0_m2_ch12;
  reg [4:0] addr_buf_1_m2_ch12;
  reg [15:0] din_buf_1_m2_ch12=0;
  wire [15:0] dout_buf_1_m2_ch12;
  
  reg [4:0] addr_buf_0_m2_ch13;
  reg wr_en_buf_0_m2_ch13=0, wr_en_buf_1_m2_ch13=0, oe_buf_0_m2_ch13=0, oe_buf_1_m2_ch13=0;
  reg [15:0] din_buf_0_m2_ch13;
  wire [15:0] dout_buf_0_m2_ch13;
  reg [4:0] addr_buf_1_m2_ch13;
  reg [15:0] din_buf_1_m2_ch13=0;
  wire [15:0] dout_buf_1_m2_ch13;
  
  reg [4:0] addr_buf_0_m2_ch14;
  reg wr_en_buf_0_m2_ch14=0, wr_en_buf_1_m2_ch14=0, oe_buf_0_m2_ch14=0, oe_buf_1_m2_ch14=0;
  reg [15:0] din_buf_0_m2_ch14;
  wire [15:0] dout_buf_0_m2_ch14;
  reg [4:0] addr_buf_1_m2_ch14;
  reg [15:0] din_buf_1_m2_ch14=0;
  wire [15:0] dout_buf_1_m2_ch14;
  
  reg [4:0] addr_buf_0_m2_ch15;
  reg wr_en_buf_0_m2_ch15=0, wr_en_buf_1_m2_ch15=0, oe_buf_0_m2_ch15=0, oe_buf_1_m2_ch15=0;
  reg [15:0] din_buf_0_m2_ch15;
  wire [15:0] dout_buf_0_m2_ch15;
  reg [4:0] addr_buf_1_m2_ch15;
  reg [15:0] din_buf_1_m2_ch15=0;
  wire [15:0] dout_buf_1_m2_ch15;
  
  reg [4:0] addr_buf_0_m2_ch16;
  reg wr_en_buf_0_m2_ch16=0, wr_en_buf_1_m2_ch16=0, oe_buf_0_m2_ch16=0, oe_buf_1_m2_ch16=0;
  reg [15:0] din_buf_0_m2_ch16;
  wire [15:0] dout_buf_0_m2_ch16;
  reg [4:0] addr_buf_1_m2_ch16;
  reg [15:0] din_buf_1_m2_ch16=0;
  wire [15:0] dout_buf_1_m2_ch16;
  
  reg [1:0]count_m1=0;
  reg done_shift_m1;
  reg en_shift_m1;
  reg rst_shift_m1;
  reg [2:0] state_m1=0;
  ///////////////////////shift window m1////////////////////////////////
  reg [7:0]count_shift_m1=0;
  reg done_mk_m1_ch1;
  reg done_mk_m1_ch2;
  reg done_mk_m1_ch3;
  reg done_mk_m1_ch4;
  reg done_mk_m1_ch5;
  reg done_mk_m1_ch6;
  reg en_mk_m1;
  reg rst_mk_m1;
  reg signed [0:15] result_temp_mk_m1_ch1;
  reg signed [0:15] result_temp_mk_m1_ch2;
  reg signed [0:15] result_temp_mk_m1_ch3;
  reg signed [0:15] result_temp_mk_m1_ch4;
  reg signed [0:15] result_temp_mk_m1_ch5;
  reg signed [0:15] result_temp_mk_m1_ch6;
  reg [2:0] state_shift_m1=0;
  ///////////////////////maxpool kernel////////////////////////////////
  wire signed [15:0]element_mk_m1_ch1[0:3];
  wire signed [15:0]element_mk_m1_ch2[0:3];
  wire signed [15:0]element_mk_m1_ch3[0:3];
  wire signed [15:0]element_mk_m1_ch4[0:3];
  wire signed [15:0]element_mk_m1_ch5[0:3];
  wire signed [15:0]element_mk_m1_ch6[0:3];
  reg signed [15:0] out_temp_mk_m1_ch1=0;
  reg signed [15:0] out_temp_mk_m1_ch2=0;
  reg signed [15:0] out_temp_mk_m1_ch3=0;
  reg signed [15:0] out_temp_mk_m1_ch4=0;
  reg signed [15:0] out_temp_mk_m1_ch5=0;
  reg signed [15:0] out_temp_mk_m1_ch6=0;
  reg [9:0] window_select_m1;
  reg done_load_mk_m1=0;
  reg [9:0] count_ld_mk_m1=0;
  wire [9:0] start_addr_mk_m1;
  reg signed [0:63] X_mk_m1_ch1;
  reg signed [0:63] X_mk_m1_ch2;
  reg signed [0:63] X_mk_m1_ch3;
  reg signed [0:63] X_mk_m1_ch4;
  reg signed [0:63] X_mk_m1_ch5;
  reg signed [0:63] X_mk_m1_ch6;
  
  reg done_load_c2d=0;
  reg [10:0] count_ld_c2d=1;
  reg signed [7:0] bias_c2d_ch1;
  reg signed [7:0] bias_c2d_ch2;
  reg signed [7:0] bias_c2d_ch3;
  reg signed [7:0] bias_c2d_ch4;
  reg signed [7:0] bias_c2d_ch5;
  reg signed [7:0] bias_c2d_ch6;
  reg signed [7:0] bias_c2d_ch7;
  reg signed [7:0] bias_c2d_ch8;
  reg signed [7:0] bias_c2d_ch9;
  reg signed [7:0] bias_c2d_ch10;
  reg signed [7:0] bias_c2d_ch11;
  reg signed [7:0] bias_c2d_ch12;
  reg signed [7:0] bias_c2d_ch13;
  reg signed [7:0] bias_c2d_ch14;
  reg signed [7:0] bias_c2d_ch15;
  reg signed [7:0] bias_c2d_ch16;
  reg signed [0:15]bias_new_c2d_ch1;
  reg signed [0:15]bias_new_c2d_ch2;
  reg signed [0:15]bias_new_c2d_ch3;
  reg signed [0:15]bias_new_c2d_ch4;
  reg signed [0:15]bias_new_c2d_ch5;
  reg signed [0:15]bias_new_c2d_ch6;
  reg signed [0:15]bias_new_c2d_ch7;
  reg signed [0:15]bias_new_c2d_ch8;
  reg signed [0:15]bias_new_c2d_ch9;
  reg signed [0:15]bias_new_c2d_ch10;
  reg signed [0:15]bias_new_c2d_ch11;
  reg signed [0:15]bias_new_c2d_ch12;
  reg signed [0:15]bias_new_c2d_ch13;
  reg signed [0:15]bias_new_c2d_ch14;
  reg signed [0:15]bias_new_c2d_ch15;
  reg signed [0:15]bias_new_c2d_ch16;
  reg [1:0]count_c2d=0;
  reg done_shift_c2d;
  reg en_shift_c2d;
  reg rst_shift_c2d;
  reg [2:0] state_c2d=0;
  //////////////////////////////////////////////////////////// shift window c2 //////////////////////////////////////////
  reg done_load_shift_c2d=0;
  reg [10:0] count_ld_shift_c2d=0;
  reg signed [0:71] W_shift_c2d_w11;
  reg signed [0:71] W_shift_c2d_w12;
  reg signed [0:71] W_shift_c2d_w13;
  reg signed [0:71] W_shift_c2d_w14;
  reg signed [0:71] W_shift_c2d_w15;
  reg signed [0:71] W_shift_c2d_w16;
  reg signed [0:71] W_shift_c2d_w21;
  reg signed [0:71] W_shift_c2d_w22;
  reg signed [0:71] W_shift_c2d_w23;
  reg signed [0:71] W_shift_c2d_w24;
  reg signed [0:71] W_shift_c2d_w25;
  reg signed [0:71] W_shift_c2d_w26;
  reg signed [0:71] W_shift_c2d_w31;
  reg signed [0:71] W_shift_c2d_w32;
  reg signed [0:71] W_shift_c2d_w33;
  reg signed [0:71] W_shift_c2d_w34;
  reg signed [0:71] W_shift_c2d_w35;
  reg signed [0:71] W_shift_c2d_w36;
  reg signed [0:71] W_shift_c2d_w41;
  reg signed [0:71] W_shift_c2d_w42;
  reg signed [0:71] W_shift_c2d_w43;
  reg signed [0:71] W_shift_c2d_w44;
  reg signed [0:71] W_shift_c2d_w45;
  reg signed [0:71] W_shift_c2d_w46;
  reg signed [0:71] W_shift_c2d_w51;
  reg signed [0:71] W_shift_c2d_w52;
  reg signed [0:71] W_shift_c2d_w53;
  reg signed [0:71] W_shift_c2d_w54;
  reg signed [0:71] W_shift_c2d_w55;
  reg signed [0:71] W_shift_c2d_w56;
  reg signed [0:71] W_shift_c2d_w61;
  reg signed [0:71] W_shift_c2d_w62;
  reg signed [0:71] W_shift_c2d_w63;
  reg signed [0:71] W_shift_c2d_w64;
  reg signed [0:71] W_shift_c2d_w65;
  reg signed [0:71] W_shift_c2d_w66;
  reg signed [0:71] W_shift_c2d_w71;
  reg signed [0:71] W_shift_c2d_w72;
  reg signed [0:71] W_shift_c2d_w73;
  reg signed [0:71] W_shift_c2d_w74;
  reg signed [0:71] W_shift_c2d_w75;
  reg signed [0:71] W_shift_c2d_w76;
  reg signed [0:71] W_shift_c2d_w81;
  reg signed [0:71] W_shift_c2d_w82;
  reg signed [0:71] W_shift_c2d_w83;
  reg signed [0:71] W_shift_c2d_w84;
  reg signed [0:71] W_shift_c2d_w85;
  reg signed [0:71] W_shift_c2d_w86;
  reg signed [0:71] W_shift_c2d_w91;
  reg signed [0:71] W_shift_c2d_w92;
  reg signed [0:71] W_shift_c2d_w93;
  reg signed [0:71] W_shift_c2d_w94;
  reg signed [0:71] W_shift_c2d_w95;
  reg signed [0:71] W_shift_c2d_w96;
  reg signed [0:71] W_shift_c2d_w101;
  reg signed [0:71] W_shift_c2d_w102;
  reg signed [0:71] W_shift_c2d_w103;
  reg signed [0:71] W_shift_c2d_w104;
  reg signed [0:71] W_shift_c2d_w105;
  reg signed [0:71] W_shift_c2d_w106;
  reg signed [0:71] W_shift_c2d_w111;
  reg signed [0:71] W_shift_c2d_w112;
  reg signed [0:71] W_shift_c2d_w113;
  reg signed [0:71] W_shift_c2d_w114;
  reg signed [0:71] W_shift_c2d_w115;
  reg signed [0:71] W_shift_c2d_w116;
  reg signed [0:71] W_shift_c2d_w121;
  reg signed [0:71] W_shift_c2d_w122;
  reg signed [0:71] W_shift_c2d_w123;
  reg signed [0:71] W_shift_c2d_w124;
  reg signed [0:71] W_shift_c2d_w125;
  reg signed [0:71] W_shift_c2d_w126;
  reg signed [0:71] W_shift_c2d_w131;
  reg signed [0:71] W_shift_c2d_w132;
  reg signed [0:71] W_shift_c2d_w133;
  reg signed [0:71] W_shift_c2d_w134;
  reg signed [0:71] W_shift_c2d_w135;
  reg signed [0:71] W_shift_c2d_w136;
  reg signed [0:71] W_shift_c2d_w141;
  reg signed [0:71] W_shift_c2d_w142;
  reg signed [0:71] W_shift_c2d_w143;
  reg signed [0:71] W_shift_c2d_w144;
  reg signed [0:71] W_shift_c2d_w145;
  reg signed [0:71] W_shift_c2d_w146;
  reg signed [0:71] W_shift_c2d_w151;
  reg signed [0:71] W_shift_c2d_w152;
  reg signed [0:71] W_shift_c2d_w153;
  reg signed [0:71] W_shift_c2d_w154;
  reg signed [0:71] W_shift_c2d_w155;
  reg signed [0:71] W_shift_c2d_w156;
  reg signed [0:71] W_shift_c2d_w161;
  reg signed [0:71] W_shift_c2d_w162;
  reg signed [0:71] W_shift_c2d_w163;
  reg signed [0:71] W_shift_c2d_w164;
  reg signed [0:71] W_shift_c2d_w165;
  reg signed [0:71] W_shift_c2d_w166;
  reg [6:0]count_shift_c2d=0;
  reg done_ck_c2d;
  reg done_ck_c2d_row_ch1;
  reg done_ck_c2d_row_ch2;
  reg done_ck_c2d_row_ch3;
  reg done_ck_c2d_row_ch4;
  reg done_ck_c2d_row_ch5;
  reg done_ck_c2d_row_ch6;
  reg done_ck_c2d_row_ch7;
  reg done_ck_c2d_row_ch8;
  reg done_ck_c2d_row_ch9;
  reg done_ck_c2d_row_ch10;
  reg done_ck_c2d_row_ch11;
  reg done_ck_c2d_row_ch12;
  reg done_ck_c2d_row_ch13;
  reg done_ck_c2d_row_ch14;
  reg done_ck_c2d_row_ch15;
  reg done_ck_c2d_row_ch16;
  reg en_ck_c2d;
  reg rst_ck_c2d;
  reg signed [0:15] result_temp_ck_c2d_in1_w11;
  reg signed [0:15] result_temp_ck_c2d_in2_w12;
  reg signed [0:15] result_temp_ck_c2d_in3_w13;
  reg signed [0:15] result_temp_ck_c2d_in4_w14;
  reg signed [0:15] result_temp_ck_c2d_in5_w15;
  reg signed [0:15] result_temp_ck_c2d_in6_w16;
  reg signed [0:15] result_temp_ck_c2d_in1_w21;
  reg signed [0:15] result_temp_ck_c2d_in2_w22;
  reg signed [0:15] result_temp_ck_c2d_in3_w23;
  reg signed [0:15] result_temp_ck_c2d_in4_w24;
  reg signed [0:15] result_temp_ck_c2d_in5_w25;
  reg signed [0:15] result_temp_ck_c2d_in6_w26;
  reg signed [0:15] result_temp_ck_c2d_in1_w31;
  reg signed [0:15] result_temp_ck_c2d_in2_w32;
  reg signed [0:15] result_temp_ck_c2d_in3_w33;
  reg signed [0:15] result_temp_ck_c2d_in4_w34;
  reg signed [0:15] result_temp_ck_c2d_in5_w35;
  reg signed [0:15] result_temp_ck_c2d_in6_w36;
  reg signed [0:15] result_temp_ck_c2d_in1_w41;
  reg signed [0:15] result_temp_ck_c2d_in2_w42;
  reg signed [0:15] result_temp_ck_c2d_in3_w43;
  reg signed [0:15] result_temp_ck_c2d_in4_w44;
  reg signed [0:15] result_temp_ck_c2d_in5_w45;
  reg signed [0:15] result_temp_ck_c2d_in6_w46;
  reg signed [0:15] result_temp_ck_c2d_in1_w51;
  reg signed [0:15] result_temp_ck_c2d_in2_w52;
  reg signed [0:15] result_temp_ck_c2d_in3_w53;
  reg signed [0:15] result_temp_ck_c2d_in4_w54;
  reg signed [0:15] result_temp_ck_c2d_in5_w55;
  reg signed [0:15] result_temp_ck_c2d_in6_w56;
  reg signed [0:15] result_temp_ck_c2d_in1_w61;
  reg signed [0:15] result_temp_ck_c2d_in2_w62;
  reg signed [0:15] result_temp_ck_c2d_in3_w63;
  reg signed [0:15] result_temp_ck_c2d_in4_w64;
  reg signed [0:15] result_temp_ck_c2d_in5_w65;
  reg signed [0:15] result_temp_ck_c2d_in6_w66;
  reg signed [0:15] result_temp_ck_c2d_in1_w71;
  reg signed [0:15] result_temp_ck_c2d_in2_w72;
  reg signed [0:15] result_temp_ck_c2d_in3_w73;
  reg signed [0:15] result_temp_ck_c2d_in4_w74;
  reg signed [0:15] result_temp_ck_c2d_in5_w75;
  reg signed [0:15] result_temp_ck_c2d_in6_w76;
  reg signed [0:15] result_temp_ck_c2d_in1_w81;
  reg signed [0:15] result_temp_ck_c2d_in2_w82;
  reg signed [0:15] result_temp_ck_c2d_in3_w83;
  reg signed [0:15] result_temp_ck_c2d_in4_w84;
  reg signed [0:15] result_temp_ck_c2d_in5_w85;
  reg signed [0:15] result_temp_ck_c2d_in6_w86;
  reg signed [0:15] result_temp_ck_c2d_in1_w91;
  reg signed [0:15] result_temp_ck_c2d_in2_w92;
  reg signed [0:15] result_temp_ck_c2d_in3_w93;
  reg signed [0:15] result_temp_ck_c2d_in4_w94;
  reg signed [0:15] result_temp_ck_c2d_in5_w95;
  reg signed [0:15] result_temp_ck_c2d_in6_w96;
  reg signed [0:15] result_temp_ck_c2d_in1_w101;
  reg signed [0:15] result_temp_ck_c2d_in2_w102;
  reg signed [0:15] result_temp_ck_c2d_in3_w103;
  reg signed [0:15] result_temp_ck_c2d_in4_w104;
  reg signed [0:15] result_temp_ck_c2d_in5_w105;
  reg signed [0:15] result_temp_ck_c2d_in6_w106;
  reg signed [0:15] result_temp_ck_c2d_in1_w111;
  reg signed [0:15] result_temp_ck_c2d_in2_w112;
  reg signed [0:15] result_temp_ck_c2d_in3_w113;
  reg signed [0:15] result_temp_ck_c2d_in4_w114;
  reg signed [0:15] result_temp_ck_c2d_in5_w115;
  reg signed [0:15] result_temp_ck_c2d_in6_w116;
  reg signed [0:15] result_temp_ck_c2d_in1_w121;
  reg signed [0:15] result_temp_ck_c2d_in2_w122;
  reg signed [0:15] result_temp_ck_c2d_in3_w123;
  reg signed [0:15] result_temp_ck_c2d_in4_w124;
  reg signed [0:15] result_temp_ck_c2d_in5_w125;
  reg signed [0:15] result_temp_ck_c2d_in6_w126;
  reg signed [0:15] result_temp_ck_c2d_in1_w131;
  reg signed [0:15] result_temp_ck_c2d_in2_w132;
  reg signed [0:15] result_temp_ck_c2d_in3_w133;
  reg signed [0:15] result_temp_ck_c2d_in4_w134;
  reg signed [0:15] result_temp_ck_c2d_in5_w135;
  reg signed [0:15] result_temp_ck_c2d_in6_w136;
  reg signed [0:15] result_temp_ck_c2d_in1_w141;
  reg signed [0:15] result_temp_ck_c2d_in2_w142;
  reg signed [0:15] result_temp_ck_c2d_in3_w143;
  reg signed [0:15] result_temp_ck_c2d_in4_w144;
  reg signed [0:15] result_temp_ck_c2d_in5_w145;
  reg signed [0:15] result_temp_ck_c2d_in6_w146;
  reg signed [0:15] result_temp_ck_c2d_in1_w151;
  reg signed [0:15] result_temp_ck_c2d_in2_w152;
  reg signed [0:15] result_temp_ck_c2d_in3_w153;
  reg signed [0:15] result_temp_ck_c2d_in4_w154;
  reg signed [0:15] result_temp_ck_c2d_in5_w155;
  reg signed [0:15] result_temp_ck_c2d_in6_w156;
  reg signed [0:15] result_temp_ck_c2d_in1_w161;
  reg signed [0:15] result_temp_ck_c2d_in2_w162;
  reg signed [0:15] result_temp_ck_c2d_in3_w163;
  reg signed [0:15] result_temp_ck_c2d_in4_w164;
  reg signed [0:15] result_temp_ck_c2d_in5_w165;
  reg signed [0:15] result_temp_ck_c2d_in6_w166;
  reg [2:0] state_shift_c2d=0;
  reg [3:0]count_ck_c2d=0;
  wire done_m_c2d_in1_w11;
  wire done_m_c2d_in2_w12;
  wire done_m_c2d_in3_w13;
  wire done_m_c2d_in4_w14;
  wire done_m_c2d_in5_w15;
  wire done_m_c2d_in6_w16;
  wire done_m_c2d_in1_w21;
  wire done_m_c2d_in2_w22;
  wire done_m_c2d_in3_w23;
  wire done_m_c2d_in4_w24;
  wire done_m_c2d_in5_w25;
  wire done_m_c2d_in6_w26;
  wire done_m_c2d_in1_w31;
  wire done_m_c2d_in2_w32;
  wire done_m_c2d_in3_w33;
  wire done_m_c2d_in4_w34;
  wire done_m_c2d_in5_w35;
  wire done_m_c2d_in6_w36;
  wire done_m_c2d_in1_w41;
  wire done_m_c2d_in2_w42;
  wire done_m_c2d_in3_w43;
  wire done_m_c2d_in4_w44;
  wire done_m_c2d_in5_w45;
  wire done_m_c2d_in6_w46;
  wire done_m_c2d_in1_w51;
  wire done_m_c2d_in2_w52;
  wire done_m_c2d_in3_w53;
  wire done_m_c2d_in4_w54;
  wire done_m_c2d_in5_w55;
  wire done_m_c2d_in6_w56;
  wire done_m_c2d_in1_w61;
  wire done_m_c2d_in2_w62;
  wire done_m_c2d_in3_w63;
  wire done_m_c2d_in4_w64;
  wire done_m_c2d_in5_w65;
  wire done_m_c2d_in6_w66;
  wire done_m_c2d_in1_w71;
  wire done_m_c2d_in2_w72;
  wire done_m_c2d_in3_w73;
  wire done_m_c2d_in4_w74;
  wire done_m_c2d_in5_w75;
  wire done_m_c2d_in6_w76;
  wire done_m_c2d_in1_w81;
  wire done_m_c2d_in2_w82;
  wire done_m_c2d_in3_w83;
  wire done_m_c2d_in4_w84;
  wire done_m_c2d_in5_w85;
  wire done_m_c2d_in6_w86;
  wire done_m_c2d_in1_w91;
  wire done_m_c2d_in2_w92;
  wire done_m_c2d_in3_w93;
  wire done_m_c2d_in4_w94;
  wire done_m_c2d_in5_w95;
  wire done_m_c2d_in6_w96;
  wire done_m_c2d_in1_w101;
  wire done_m_c2d_in2_w102;
  wire done_m_c2d_in3_w103;
  wire done_m_c2d_in4_w104;
  wire done_m_c2d_in5_w105;
  wire done_m_c2d_in6_w106;
  wire done_m_c2d_in1_w111;
  wire done_m_c2d_in2_w112;
  wire done_m_c2d_in3_w113;
  wire done_m_c2d_in4_w114;
  wire done_m_c2d_in5_w115;
  wire done_m_c2d_in6_w116;
  wire done_m_c2d_in1_w121;
  wire done_m_c2d_in2_w122;
  wire done_m_c2d_in3_w123;
  wire done_m_c2d_in4_w124;
  wire done_m_c2d_in5_w125;
  wire done_m_c2d_in6_w126;
  wire done_m_c2d_in1_w131;
  wire done_m_c2d_in2_w132;
  wire done_m_c2d_in3_w133;
  wire done_m_c2d_in4_w134;
  wire done_m_c2d_in5_w135;
  wire done_m_c2d_in6_w136;
  wire done_m_c2d_in1_w141;
  wire done_m_c2d_in2_w142;
  wire done_m_c2d_in3_w143;
  wire done_m_c2d_in4_w144;
  wire done_m_c2d_in5_w145;
  wire done_m_c2d_in6_w146;
  wire done_m_c2d_in1_w151;
  wire done_m_c2d_in2_w152;
  wire done_m_c2d_in3_w153;
  wire done_m_c2d_in4_w154;
  wire done_m_c2d_in5_w155;
  wire done_m_c2d_in6_w156;
  wire done_m_c2d_in1_w161;
  wire done_m_c2d_in2_w162;
  wire done_m_c2d_in3_w163;
  wire done_m_c2d_in4_w164;
  wire done_m_c2d_in5_w165;
  wire done_m_c2d_in6_w166;
  reg en_m_c2d;
  reg rst_m_c2d;
  wire signed [31:0] result_temp_m_c2d_in1_w11;
  wire signed [31:0] result_temp_m_c2d_in2_w12;
  wire signed [31:0] result_temp_m_c2d_in3_w13;
  wire signed [31:0] result_temp_m_c2d_in4_w14;
  wire signed [31:0] result_temp_m_c2d_in5_w15;
  wire signed [31:0] result_temp_m_c2d_in6_w16;
  wire signed [31:0] result_temp_m_c2d_in1_w21;
  wire signed [31:0] result_temp_m_c2d_in2_w22;
  wire signed [31:0] result_temp_m_c2d_in3_w23;
  wire signed [31:0] result_temp_m_c2d_in4_w24;
  wire signed [31:0] result_temp_m_c2d_in5_w25;
  wire signed [31:0] result_temp_m_c2d_in6_w26;
  wire signed [31:0] result_temp_m_c2d_in1_w31;
  wire signed [31:0] result_temp_m_c2d_in2_w32;
  wire signed [31:0] result_temp_m_c2d_in3_w33;
  wire signed [31:0] result_temp_m_c2d_in4_w34;
  wire signed [31:0] result_temp_m_c2d_in5_w35;
  wire signed [31:0] result_temp_m_c2d_in6_w36;
  wire signed [31:0] result_temp_m_c2d_in1_w41;
  wire signed [31:0] result_temp_m_c2d_in2_w42;
  wire signed [31:0] result_temp_m_c2d_in3_w43;
  wire signed [31:0] result_temp_m_c2d_in4_w44;
  wire signed [31:0] result_temp_m_c2d_in5_w45;
  wire signed [31:0] result_temp_m_c2d_in6_w46;
  wire signed [31:0] result_temp_m_c2d_in1_w51;
  wire signed [31:0] result_temp_m_c2d_in2_w52;
  wire signed [31:0] result_temp_m_c2d_in3_w53;
  wire signed [31:0] result_temp_m_c2d_in4_w54;
  wire signed [31:0] result_temp_m_c2d_in5_w55;
  wire signed [31:0] result_temp_m_c2d_in6_w56;
  wire signed [31:0] result_temp_m_c2d_in1_w61;
  wire signed [31:0] result_temp_m_c2d_in2_w62;
  wire signed [31:0] result_temp_m_c2d_in3_w63;
  wire signed [31:0] result_temp_m_c2d_in4_w64;
  wire signed [31:0] result_temp_m_c2d_in5_w65;
  wire signed [31:0] result_temp_m_c2d_in6_w66;
  wire signed [31:0] result_temp_m_c2d_in1_w71;
  wire signed [31:0] result_temp_m_c2d_in2_w72;
  wire signed [31:0] result_temp_m_c2d_in3_w73;
  wire signed [31:0] result_temp_m_c2d_in4_w74;
  wire signed [31:0] result_temp_m_c2d_in5_w75;
  wire signed [31:0] result_temp_m_c2d_in6_w76;
  wire signed [31:0] result_temp_m_c2d_in1_w81;
  wire signed [31:0] result_temp_m_c2d_in2_w82;
  wire signed [31:0] result_temp_m_c2d_in3_w83;
  wire signed [31:0] result_temp_m_c2d_in4_w84;
  wire signed [31:0] result_temp_m_c2d_in5_w85;
  wire signed [31:0] result_temp_m_c2d_in6_w86;
  wire signed [31:0] result_temp_m_c2d_in1_w91;
  wire signed [31:0] result_temp_m_c2d_in2_w92;
  wire signed [31:0] result_temp_m_c2d_in3_w93;
  wire signed [31:0] result_temp_m_c2d_in4_w94;
  wire signed [31:0] result_temp_m_c2d_in5_w95;
  wire signed [31:0] result_temp_m_c2d_in6_w96;
  wire signed [31:0] result_temp_m_c2d_in1_w101;
  wire signed [31:0] result_temp_m_c2d_in2_w102;
  wire signed [31:0] result_temp_m_c2d_in3_w103;
  wire signed [31:0] result_temp_m_c2d_in4_w104;
  wire signed [31:0] result_temp_m_c2d_in5_w105;
  wire signed [31:0] result_temp_m_c2d_in6_w106;
  wire signed [31:0] result_temp_m_c2d_in1_w111;
  wire signed [31:0] result_temp_m_c2d_in2_w112;
  wire signed [31:0] result_temp_m_c2d_in3_w113;
  wire signed [31:0] result_temp_m_c2d_in4_w114;
  wire signed [31:0] result_temp_m_c2d_in5_w115;
  wire signed [31:0] result_temp_m_c2d_in6_w116;
  wire signed [31:0] result_temp_m_c2d_in1_w121;
  wire signed [31:0] result_temp_m_c2d_in2_w122;
  wire signed [31:0] result_temp_m_c2d_in3_w123;
  wire signed [31:0] result_temp_m_c2d_in4_w124;
  wire signed [31:0] result_temp_m_c2d_in5_w125;
  wire signed [31:0] result_temp_m_c2d_in6_w126;
  wire signed [31:0] result_temp_m_c2d_in1_w131;
  wire signed [31:0] result_temp_m_c2d_in2_w132;
  wire signed [31:0] result_temp_m_c2d_in3_w133;
  wire signed [31:0] result_temp_m_c2d_in4_w134;
  wire signed [31:0] result_temp_m_c2d_in5_w135;
  wire signed [31:0] result_temp_m_c2d_in6_w136;
  wire signed [31:0] result_temp_m_c2d_in1_w141;
  wire signed [31:0] result_temp_m_c2d_in2_w142;
  wire signed [31:0] result_temp_m_c2d_in3_w143;
  wire signed [31:0] result_temp_m_c2d_in4_w144;
  wire signed [31:0] result_temp_m_c2d_in5_w145;
  wire signed [31:0] result_temp_m_c2d_in6_w146;
  wire signed [31:0] result_temp_m_c2d_in1_w151;
  wire signed [31:0] result_temp_m_c2d_in2_w152;
  wire signed [31:0] result_temp_m_c2d_in3_w153;
  wire signed [31:0] result_temp_m_c2d_in4_w154;
  wire signed [31:0] result_temp_m_c2d_in5_w155;
  wire signed [31:0] result_temp_m_c2d_in6_w156;
  wire signed [31:0] result_temp_m_c2d_in1_w161;
  wire signed [31:0] result_temp_m_c2d_in2_w162;
  wire signed [31:0] result_temp_m_c2d_in3_w163;
  wire signed [31:0] result_temp_m_c2d_in4_w164;
  wire signed [31:0] result_temp_m_c2d_in5_w165;
  wire signed [31:0] result_temp_m_c2d_in6_w166;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w11;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w12;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w13;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w14;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w15;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w16;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w21;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w22;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w23;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w24;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w25;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w26;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w31;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w32;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w33;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w34;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w35;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w36;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w41;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w42;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w43;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w44;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w45;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w46;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w51;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w52;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w53;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w54;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w55;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w56;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w61;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w62;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w63;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w64;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w65;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w66;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w71;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w72;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w73;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w74;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w75;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w76;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w81;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w82;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w83;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w84;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w85;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w86;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w91;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w92;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w93;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w94;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w95;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w96;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w101;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w102;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w103;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w104;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w105;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w106;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w111;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w112;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w113;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w114;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w115;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w116;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w121;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w122;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w123;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w124;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w125;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w126;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w131;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w132;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w133;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w134;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w135;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w136;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w141;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w142;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w143;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w144;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w145;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w146;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w151;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w152;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w153;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w154;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w155;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w156;
  reg signed [31:0] result_final_temp_ck_c2d_in1_w161;
  reg signed [31:0] result_final_temp_ck_c2d_in2_w162;
  reg signed [31:0] result_final_temp_ck_c2d_in3_w163;
  reg signed [31:0] result_final_temp_ck_c2d_in4_w164;
  reg signed [31:0] result_final_temp_ck_c2d_in5_w165;
  reg signed [31:0] result_final_temp_ck_c2d_in6_w166;
  reg signed [31:0] buffer_ck_c2d_in1_w11=0;
  reg signed [31:0] buffer_ck_c2d_in2_w12=0;
  reg signed [31:0] buffer_ck_c2d_in3_w13=0;
  reg signed [31:0] buffer_ck_c2d_in4_w14=0;
  reg signed [31:0] buffer_ck_c2d_in5_w15=0;
  reg signed [31:0] buffer_ck_c2d_in6_w16=0;
  reg signed [31:0] buffer_ck_c2d_in1_w21=0;
  reg signed [31:0] buffer_ck_c2d_in2_w22=0;
  reg signed [31:0] buffer_ck_c2d_in3_w23=0;
  reg signed [31:0] buffer_ck_c2d_in4_w24=0;
  reg signed [31:0] buffer_ck_c2d_in5_w25=0;
  reg signed [31:0] buffer_ck_c2d_in6_w26=0;
  reg signed [31:0] buffer_ck_c2d_in1_w31=0;
  reg signed [31:0] buffer_ck_c2d_in2_w32=0;
  reg signed [31:0] buffer_ck_c2d_in3_w33=0;
  reg signed [31:0] buffer_ck_c2d_in4_w34=0;
  reg signed [31:0] buffer_ck_c2d_in5_w35=0;
  reg signed [31:0] buffer_ck_c2d_in6_w36=0;
  reg signed [31:0] buffer_ck_c2d_in1_w41=0;
  reg signed [31:0] buffer_ck_c2d_in2_w42=0;
  reg signed [31:0] buffer_ck_c2d_in3_w43=0;
  reg signed [31:0] buffer_ck_c2d_in4_w44=0;
  reg signed [31:0] buffer_ck_c2d_in5_w45=0;
  reg signed [31:0] buffer_ck_c2d_in6_w46=0;
  reg signed [31:0] buffer_ck_c2d_in1_w51=0;
  reg signed [31:0] buffer_ck_c2d_in2_w52=0;
  reg signed [31:0] buffer_ck_c2d_in3_w53=0;
  reg signed [31:0] buffer_ck_c2d_in4_w54=0;
  reg signed [31:0] buffer_ck_c2d_in5_w55=0;
  reg signed [31:0] buffer_ck_c2d_in6_w56=0;
  reg signed [31:0] buffer_ck_c2d_in1_w61=0;
  reg signed [31:0] buffer_ck_c2d_in2_w62=0;
  reg signed [31:0] buffer_ck_c2d_in3_w63=0;
  reg signed [31:0] buffer_ck_c2d_in4_w64=0;
  reg signed [31:0] buffer_ck_c2d_in5_w65=0;
  reg signed [31:0] buffer_ck_c2d_in6_w66=0;
  reg signed [31:0] buffer_ck_c2d_in1_w71=0;
  reg signed [31:0] buffer_ck_c2d_in2_w72=0;
  reg signed [31:0] buffer_ck_c2d_in3_w73=0;
  reg signed [31:0] buffer_ck_c2d_in4_w74=0;
  reg signed [31:0] buffer_ck_c2d_in5_w75=0;
  reg signed [31:0] buffer_ck_c2d_in6_w76=0;
  reg signed [31:0] buffer_ck_c2d_in1_w81=0;
  reg signed [31:0] buffer_ck_c2d_in2_w82=0;
  reg signed [31:0] buffer_ck_c2d_in3_w83=0;
  reg signed [31:0] buffer_ck_c2d_in4_w84=0;
  reg signed [31:0] buffer_ck_c2d_in5_w85=0;
  reg signed [31:0] buffer_ck_c2d_in6_w86=0;
  reg signed [31:0] buffer_ck_c2d_in1_w91=0;
  reg signed [31:0] buffer_ck_c2d_in2_w92=0;
  reg signed [31:0] buffer_ck_c2d_in3_w93=0;
  reg signed [31:0] buffer_ck_c2d_in4_w94=0;
  reg signed [31:0] buffer_ck_c2d_in5_w95=0;
  reg signed [31:0] buffer_ck_c2d_in6_w96=0;
  reg signed [31:0] buffer_ck_c2d_in1_w101=0;
  reg signed [31:0] buffer_ck_c2d_in2_w102=0;
  reg signed [31:0] buffer_ck_c2d_in3_w103=0;
  reg signed [31:0] buffer_ck_c2d_in4_w104=0;
  reg signed [31:0] buffer_ck_c2d_in5_w105=0;
  reg signed [31:0] buffer_ck_c2d_in6_w106=0;
  reg signed [31:0] buffer_ck_c2d_in1_w111=0;
  reg signed [31:0] buffer_ck_c2d_in2_w112=0;
  reg signed [31:0] buffer_ck_c2d_in3_w113=0;
  reg signed [31:0] buffer_ck_c2d_in4_w114=0;
  reg signed [31:0] buffer_ck_c2d_in5_w115=0;
  reg signed [31:0] buffer_ck_c2d_in6_w116=0;
  reg signed [31:0] buffer_ck_c2d_in1_w121=0;
  reg signed [31:0] buffer_ck_c2d_in2_w122=0;
  reg signed [31:0] buffer_ck_c2d_in3_w123=0;
  reg signed [31:0] buffer_ck_c2d_in4_w124=0;
  reg signed [31:0] buffer_ck_c2d_in5_w125=0;
  reg signed [31:0] buffer_ck_c2d_in6_w126=0;
  reg signed [31:0] buffer_ck_c2d_in1_w131=0;
  reg signed [31:0] buffer_ck_c2d_in2_w132=0;
  reg signed [31:0] buffer_ck_c2d_in3_w133=0;
  reg signed [31:0] buffer_ck_c2d_in4_w134=0;
  reg signed [31:0] buffer_ck_c2d_in5_w135=0;
  reg signed [31:0] buffer_ck_c2d_in6_w136=0;
  reg signed [31:0] buffer_ck_c2d_in1_w141=0;
  reg signed [31:0] buffer_ck_c2d_in2_w142=0;
  reg signed [31:0] buffer_ck_c2d_in3_w143=0;
  reg signed [31:0] buffer_ck_c2d_in4_w144=0;
  reg signed [31:0] buffer_ck_c2d_in5_w145=0;
  reg signed [31:0] buffer_ck_c2d_in6_w146=0;
  reg signed [31:0] buffer_ck_c2d_in1_w151=0;
  reg signed [31:0] buffer_ck_c2d_in2_w152=0;
  reg signed [31:0] buffer_ck_c2d_in3_w153=0;
  reg signed [31:0] buffer_ck_c2d_in4_w154=0;
  reg signed [31:0] buffer_ck_c2d_in5_w155=0;
  reg signed [31:0] buffer_ck_c2d_in6_w156=0;
  reg signed [31:0] buffer_ck_c2d_in1_w161=0;
  reg signed [31:0] buffer_ck_c2d_in2_w162=0;
  reg signed [31:0] buffer_ck_c2d_in3_w163=0;
  reg signed [31:0] buffer_ck_c2d_in4_w164=0;
  reg signed [31:0] buffer_ck_c2d_in5_w165=0;
  reg signed [31:0] buffer_ck_c2d_in6_w166=0;
  wire signed [15:0] mem_x_ck_c2d_ch_in1 [0:8];
  wire signed [15:0] mem_x_ck_c2d_ch_in2 [0:8];
  wire signed [15:0] mem_x_ck_c2d_ch_in3 [0:8];
  wire signed [15:0] mem_x_ck_c2d_ch_in4 [0:8];
  wire signed [15:0] mem_x_ck_c2d_ch_in5 [0:8];
  wire signed [15:0] mem_x_ck_c2d_ch_in6 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w11 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w12 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w13 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w14 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w15 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w16 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w21 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w22 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w23 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w24 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w25 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w26 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w31 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w32 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w33 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w34 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w35 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w36 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w41 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w42 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w43 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w44 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w45 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w46 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w51 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w52 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w53 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w54 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w55 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w56 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w61 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w62 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w63 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w64 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w65 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w66 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w71 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w72 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w73 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w74 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w75 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w76 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w81 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w82 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w83 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w84 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w85 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w86 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w91 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w92 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w93 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w94 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w95 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w96 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w101 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w102 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w103 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w104 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w105 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w106 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w111 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w112 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w113 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w114 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w115 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w116 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w121 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w122 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w123 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w124 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w125 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w126 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w131 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w132 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w133 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w134 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w135 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w136 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w141 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w142 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w143 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w144 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w145 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w146 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w151 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w152 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w153 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w154 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w155 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w156 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w161 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w162 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w163 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w164 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w165 [0:8];
  wire signed [7:0] mem_w_ck_c2d_w166 [0:8];
  reg signed [15:0] x_in_m_c2d_ch_in1;
  reg signed [15:0] x_in_m_c2d_ch_in2;
  reg signed [15:0] x_in_m_c2d_ch_in3;
  reg signed [15:0] x_in_m_c2d_ch_in4;
  reg signed [15:0] x_in_m_c2d_ch_in5;
  reg signed [15:0] x_in_m_c2d_ch_in6;
  reg signed [7:0] w_in_m_c2d_w11;
  reg signed [7:0] w_in_m_c2d_w12;
  reg signed [7:0] w_in_m_c2d_w13;
  reg signed [7:0] w_in_m_c2d_w14;
  reg signed [7:0] w_in_m_c2d_w15;
  reg signed [7:0] w_in_m_c2d_w16;
  reg signed [7:0] w_in_m_c2d_w21;
  reg signed [7:0] w_in_m_c2d_w22;
  reg signed [7:0] w_in_m_c2d_w23;
  reg signed [7:0] w_in_m_c2d_w24;
  reg signed [7:0] w_in_m_c2d_w25;
  reg signed [7:0] w_in_m_c2d_w26;
  reg signed [7:0] w_in_m_c2d_w31;
  reg signed [7:0] w_in_m_c2d_w32;
  reg signed [7:0] w_in_m_c2d_w33;
  reg signed [7:0] w_in_m_c2d_w34;
  reg signed [7:0] w_in_m_c2d_w35;
  reg signed [7:0] w_in_m_c2d_w36;
  reg signed [7:0] w_in_m_c2d_w41;
  reg signed [7:0] w_in_m_c2d_w42;
  reg signed [7:0] w_in_m_c2d_w43;
  reg signed [7:0] w_in_m_c2d_w44;
  reg signed [7:0] w_in_m_c2d_w45;
  reg signed [7:0] w_in_m_c2d_w46;
  reg signed [7:0] w_in_m_c2d_w51;
  reg signed [7:0] w_in_m_c2d_w52;
  reg signed [7:0] w_in_m_c2d_w53;
  reg signed [7:0] w_in_m_c2d_w54;
  reg signed [7:0] w_in_m_c2d_w55;
  reg signed [7:0] w_in_m_c2d_w56;
  reg signed [7:0] w_in_m_c2d_w61;
  reg signed [7:0] w_in_m_c2d_w62;
  reg signed [7:0] w_in_m_c2d_w63;
  reg signed [7:0] w_in_m_c2d_w64;
  reg signed [7:0] w_in_m_c2d_w65;
  reg signed [7:0] w_in_m_c2d_w66;
  reg signed [7:0] w_in_m_c2d_w71;
  reg signed [7:0] w_in_m_c2d_w72;
  reg signed [7:0] w_in_m_c2d_w73;
  reg signed [7:0] w_in_m_c2d_w74;
  reg signed [7:0] w_in_m_c2d_w75;
  reg signed [7:0] w_in_m_c2d_w76;
  reg signed [7:0] w_in_m_c2d_w81;
  reg signed [7:0] w_in_m_c2d_w82;
  reg signed [7:0] w_in_m_c2d_w83;
  reg signed [7:0] w_in_m_c2d_w84;
  reg signed [7:0] w_in_m_c2d_w85;
  reg signed [7:0] w_in_m_c2d_w86;
  reg signed [7:0] w_in_m_c2d_w91;
  reg signed [7:0] w_in_m_c2d_w92;
  reg signed [7:0] w_in_m_c2d_w93;
  reg signed [7:0] w_in_m_c2d_w94;
  reg signed [7:0] w_in_m_c2d_w95;
  reg signed [7:0] w_in_m_c2d_w96;
  reg signed [7:0] w_in_m_c2d_w101;
  reg signed [7:0] w_in_m_c2d_w102;
  reg signed [7:0] w_in_m_c2d_w103;
  reg signed [7:0] w_in_m_c2d_w104;
  reg signed [7:0] w_in_m_c2d_w105;
  reg signed [7:0] w_in_m_c2d_w106;
  reg signed [7:0] w_in_m_c2d_w111;
  reg signed [7:0] w_in_m_c2d_w112;
  reg signed [7:0] w_in_m_c2d_w113;
  reg signed [7:0] w_in_m_c2d_w114;
  reg signed [7:0] w_in_m_c2d_w115;
  reg signed [7:0] w_in_m_c2d_w116;
  reg signed [7:0] w_in_m_c2d_w121;
  reg signed [7:0] w_in_m_c2d_w122;
  reg signed [7:0] w_in_m_c2d_w123;
  reg signed [7:0] w_in_m_c2d_w124;
  reg signed [7:0] w_in_m_c2d_w125;
  reg signed [7:0] w_in_m_c2d_w126;
  reg signed [7:0] w_in_m_c2d_w131;
  reg signed [7:0] w_in_m_c2d_w132;
  reg signed [7:0] w_in_m_c2d_w133;
  reg signed [7:0] w_in_m_c2d_w134;
  reg signed [7:0] w_in_m_c2d_w135;
  reg signed [7:0] w_in_m_c2d_w136;
  reg signed [7:0] w_in_m_c2d_w141;
  reg signed [7:0] w_in_m_c2d_w142;
  reg signed [7:0] w_in_m_c2d_w143;
  reg signed [7:0] w_in_m_c2d_w144;
  reg signed [7:0] w_in_m_c2d_w145;
  reg signed [7:0] w_in_m_c2d_w146;
  reg signed [7:0] w_in_m_c2d_w151;
  reg signed [7:0] w_in_m_c2d_w152;
  reg signed [7:0] w_in_m_c2d_w153;
  reg signed [7:0] w_in_m_c2d_w154;
  reg signed [7:0] w_in_m_c2d_w155;
  reg signed [7:0] w_in_m_c2d_w156;
  reg signed [7:0] w_in_m_c2d_w161;
  reg signed [7:0] w_in_m_c2d_w162;
  reg signed [7:0] w_in_m_c2d_w163;
  reg signed [7:0] w_in_m_c2d_w164;
  reg signed [7:0] w_in_m_c2d_w165;
  reg signed [7:0] w_in_m_c2d_w166;
  reg [2:0] state_ck_c2d=0;
  reg done_load_conv_c2=0;
  reg [10:0] count_ld_conv_c2=0;
  wire [10:0] start_addr_conv_c2;
  reg signed [0:143] X_conv_c2_ch_in1;
  reg signed [0:143] X_conv_c2_ch_in2;
  reg signed [0:143] X_conv_c2_ch_in3;
  reg signed [0:143] X_conv_c2_ch_in4;
  reg signed [0:143] X_conv_c2_ch_in5;
  reg signed [0:143] X_conv_c2_ch_in6;
  reg [7:0] window_select_c2;
  reg [3:0] n=1, m=1, a=1;
  reg done_mk_m1_row_ch1=0;
  reg done_mk_m1_row_ch2=0;
  reg done_mk_m1_row_ch3=0;
  reg done_mk_m1_row_ch4=0;
  reg done_mk_m1_row_ch5=0;
  reg done_mk_m1_row_ch6=0;
  ////////////////////////////////////////////////// maxpool m2 ////////////////////////////////////////////////////////////////
  reg [1:0]count_m2=0;
  reg done_shift_m2;
  reg en_shift_m2;
  reg rst_shift_m2;
  reg [2:0] state_m2=0;
  /////////////////////////////////////////////////// shift window m2 ////////////////////////////////////////////////////////////////////////
  reg [4:0]count_shift_m2=0;
  reg done_mk_m2;
  reg en_mk_m2;
  reg rst_mk_m2;
  reg signed [0:31] result_temp_mk_m2_ch1;
  reg signed [0:31] result_temp_mk_m2_ch2;
  reg signed [0:31] result_temp_mk_m2_ch3;
  reg signed [0:31] result_temp_mk_m2_ch4;
  reg signed [0:31] result_temp_mk_m2_ch5;
  reg signed [0:31] result_temp_mk_m2_ch6;
  reg signed [0:31] result_temp_mk_m2_ch7;
  reg signed [0:31] result_temp_mk_m2_ch8;
  reg signed [0:31] result_temp_mk_m2_ch9;
  reg signed [0:31] result_temp_mk_m2_ch10;
  reg signed [0:31] result_temp_mk_m2_ch11;
  reg signed [0:31] result_temp_mk_m2_ch12;
  reg signed [0:31] result_temp_mk_m2_ch13;
  reg signed [0:31] result_temp_mk_m2_ch14;
  reg signed [0:31] result_temp_mk_m2_ch15;
  reg signed [0:31] result_temp_mk_m2_ch16;
  reg [2:0] state_shift_m2=0;
  ////////////////////////////////////// maxpool kernel //////////////////////////////////////////
  wire signed [15:0]element_mk_m2_ch1[0:8];
  wire signed [15:0]element_mk_m2_ch2[0:8];
  wire signed [15:0]element_mk_m2_ch3[0:8];
  wire signed [15:0]element_mk_m2_ch4[0:8];
  wire signed [15:0]element_mk_m2_ch5[0:8];
  wire signed [15:0]element_mk_m2_ch6[0:8];
  wire signed [15:0]element_mk_m2_ch7[0:8];
  wire signed [15:0]element_mk_m2_ch8[0:8];
  wire signed [15:0]element_mk_m2_ch9[0:8];
  wire signed [15:0]element_mk_m2_ch10[0:8];
  wire signed [15:0]element_mk_m2_ch11[0:8];
  wire signed [15:0]element_mk_m2_ch12[0:8];
  wire signed [15:0]element_mk_m2_ch13[0:8];
  wire signed [15:0]element_mk_m2_ch14[0:8];
  wire signed [15:0]element_mk_m2_ch15[0:8];
  wire signed [15:0]element_mk_m2_ch16[0:8];
  reg signed [31:0] out_temp_mk_m2_ch1=0;
  reg signed [31:0] out_temp_mk_m2_ch2=0;
  reg signed [31:0] out_temp_mk_m2_ch3=0;
  reg signed [31:0] out_temp_mk_m2_ch4=0;
  reg signed [31:0] out_temp_mk_m2_ch5=0;
  reg signed [31:0] out_temp_mk_m2_ch6=0;
  reg signed [31:0] out_temp_mk_m2_ch7=0;
  reg signed [31:0] out_temp_mk_m2_ch8=0;
  reg signed [31:0] out_temp_mk_m2_ch9=0;
  reg signed [31:0] out_temp_mk_m2_ch10=0;
  reg signed [31:0] out_temp_mk_m2_ch11=0;
  reg signed [31:0] out_temp_mk_m2_ch12=0;
  reg signed [31:0] out_temp_mk_m2_ch13=0;
  reg signed [31:0] out_temp_mk_m2_ch14=0;
  reg signed [31:0] out_temp_mk_m2_ch15=0;
  reg signed [31:0] out_temp_mk_m2_ch16=0;
  reg done_mk_m2_row_ch1=0;
  reg done_mk_m2_row_ch2=0;
  reg done_mk_m2_row_ch3=0;
  reg done_mk_m2_row_ch4=0;
  reg done_mk_m2_row_ch5=0;
  reg done_mk_m2_row_ch6=0;
  reg done_mk_m2_row_ch7=0;
  reg done_mk_m2_row_ch8=0;
  reg done_mk_m2_row_ch9=0;
  reg done_mk_m2_row_ch10=0;
  reg done_mk_m2_row_ch11=0;
  reg done_mk_m2_row_ch12=0;
  reg done_mk_m2_row_ch13=0;
  reg done_mk_m2_row_ch14=0;
  reg done_mk_m2_row_ch15=0;
  reg done_mk_m2_row_ch16=0;
  reg [1:0] channel_select_m2;
  reg [9:0] window_select_m2;
  reg done_load_mk_m2=0;
  reg [9:0] count_ld_mk_m2=0;
  wire [9:0] start_addr_mk_m2;
  reg signed [0:143] X_mk_m2_ch1;
  reg signed [0:143] X_mk_m2_ch2;
  reg signed [0:143] X_mk_m2_ch3;
  reg signed [0:143] X_mk_m2_ch4;
  reg signed [0:143] X_mk_m2_ch5;
  reg signed [0:143] X_mk_m2_ch6;
  reg signed [0:143] X_mk_m2_ch7;
  reg signed [0:143] X_mk_m2_ch8;
  reg signed [0:143] X_mk_m2_ch9;
  reg signed [0:143] X_mk_m2_ch10;
  reg signed [0:143] X_mk_m2_ch11;
  reg signed [0:143] X_mk_m2_ch12;
  reg signed [0:143] X_mk_m2_ch13;
  reg signed [0:143] X_mk_m2_ch14;
  reg signed [0:143] X_mk_m2_ch15;
  reg signed [0:143] X_mk_m2_ch16;
///////////////////////////////////////// dense 1 ////////////////////////////////////////
  reg [4:0] count_ld_d1=0;
  reg done_load_d1;
  reg [9:0] start_addr_d1;
  reg [0:15] X_nc_d1_ch1;
  reg [0:15] X_nc_d1_ch2;
  reg [0:15] X_nc_d1_ch3;
  reg [0:15] X_nc_d1_ch4;
  reg [0:15] X_nc_d1_ch5;
  reg [0:15] X_nc_d1_ch6;
  reg [0:15] X_nc_d1_ch7;
  reg [0:15] X_nc_d1_ch8;
  reg [0:15] X_nc_d1_ch9;
  reg [0:15] X_nc_d1_ch10;
  reg [0:15] X_nc_d1_ch11;
  reg [0:15] X_nc_d1_ch12;
  reg [0:15] X_nc_d1_ch13;
  reg [0:15] X_nc_d1_ch14;
  reg [0:15] X_nc_d1_ch15;
  reg [0:15] X_nc_d1_ch16;
  reg [3:0] neuron_select_d1;
  reg [4:0]count_d1=0;
  reg done_nc_d1;
  reg en_nc_d1;
  reg rst_nc_d1;
  reg signed [0:31] result_temp_nc_d1;
  reg [2:0] state_d1=0;
////////////////////////////////////// neuron calculation /////////////////////////////////////
  reg done_load_nc_d1=0;
  reg [10:0] count_ld_nc_d1=0;
  reg signed [0:79] w_neuron_nc_d1_ch1;
  reg signed [0:79] w_neuron_nc_d1_ch2;
  reg signed [0:79] w_neuron_nc_d1_ch3;
  reg signed [0:79] w_neuron_nc_d1_ch4;
  reg signed [0:79] w_neuron_nc_d1_ch5;
  reg signed [0:79] w_neuron_nc_d1_ch6;
  reg signed [0:79] w_neuron_nc_d1_ch7;
  reg signed [0:79] w_neuron_nc_d1_ch8;
  reg signed [0:79] w_neuron_nc_d1_ch9;
  reg signed [0:79] w_neuron_nc_d1_ch10;
  reg signed [0:79] w_neuron_nc_d1_ch11;
  reg signed [0:79] w_neuron_nc_d1_ch12;
  reg signed [0:79] w_neuron_nc_d1_ch13;
  reg signed [0:79] w_neuron_nc_d1_ch14;
  reg signed [0:79] w_neuron_nc_d1_ch15;
  reg signed [0:79] w_neuron_nc_d1_ch16;
  reg signed [0:79] bias_d1;
  reg [4:0]count_nc_d1=0;
  wire done_m_nc_d1_ch1;
  wire done_m_nc_d1_ch2;
  wire done_m_nc_d1_ch3;
  wire done_m_nc_d1_ch4;
  wire done_m_nc_d1_ch5;
  wire done_m_nc_d1_ch6;
  wire done_m_nc_d1_ch7;
  wire done_m_nc_d1_ch8;
  wire done_m_nc_d1_ch9;
  wire done_m_nc_d1_ch10;
  wire done_m_nc_d1_ch11;
  wire done_m_nc_d1_ch12;
  wire done_m_nc_d1_ch13;
  wire done_m_nc_d1_ch14;
  wire done_m_nc_d1_ch15;
  wire done_m_nc_d1_ch16;
  reg en_m_nc_d1;
  reg rst_m_nc_d1;
  wire signed [31:0] result_temp_m_nc_d1_ch1;
  wire signed [31:0] result_temp_m_nc_d1_ch2;
  wire signed [31:0] result_temp_m_nc_d1_ch3;
  wire signed [31:0] result_temp_m_nc_d1_ch4;
  wire signed [31:0] result_temp_m_nc_d1_ch5;
  wire signed [31:0] result_temp_m_nc_d1_ch6;
  wire signed [31:0] result_temp_m_nc_d1_ch7;
  wire signed [31:0] result_temp_m_nc_d1_ch8;
  wire signed [31:0] result_temp_m_nc_d1_ch9;
  wire signed [31:0] result_temp_m_nc_d1_ch10;
  wire signed [31:0] result_temp_m_nc_d1_ch11;
  wire signed [31:0] result_temp_m_nc_d1_ch12;
  wire signed [31:0] result_temp_m_nc_d1_ch13;
  wire signed [31:0] result_temp_m_nc_d1_ch14;
  wire signed [31:0] result_temp_m_nc_d1_ch15;
  wire signed [31:0] result_temp_m_nc_d1_ch16;
  reg signed [0:319] buffer_nc_d1_ch1=0;
  reg signed [0:319] buffer_nc_d1_ch2=0;
  reg signed [0:319] buffer_nc_d1_ch3=0;
  reg signed [0:319] buffer_nc_d1_ch4=0;
  reg signed [0:319] buffer_nc_d1_ch5=0;
  reg signed [0:319] buffer_nc_d1_ch6=0;
  reg signed [0:319] buffer_nc_d1_ch7=0;
  reg signed [0:319] buffer_nc_d1_ch8=0;
  reg signed [0:319] buffer_nc_d1_ch9=0;
  reg signed [0:319] buffer_nc_d1_ch10=0;
  reg signed [0:319] buffer_nc_d1_ch11=0;
  reg signed [0:319] buffer_nc_d1_ch12=0;
  reg signed [0:319] buffer_nc_d1_ch13=0;
  reg signed [0:319] buffer_nc_d1_ch14=0;
  reg signed [0:319] buffer_nc_d1_ch15=0;
  reg signed [0:319] buffer_nc_d1_ch16=0;
  wire signed [7:0] mem_w_nc_d1_ch1 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch2 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch3 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch4 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch5 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch6 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch7 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch8 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch9 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch10 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch11 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch12 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch13 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch14 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch15 [0:9];
  wire signed [7:0] mem_w_nc_d1_ch16 [0:9];
  reg signed [7:0] w_in_m_nc_d1_ch1;
  reg signed [7:0] w_in_m_nc_d1_ch2;
  reg signed [7:0] w_in_m_nc_d1_ch3;
  reg signed [7:0] w_in_m_nc_d1_ch4;
  reg signed [7:0] w_in_m_nc_d1_ch5;
  reg signed [7:0] w_in_m_nc_d1_ch6;
  reg signed [7:0] w_in_m_nc_d1_ch7;
  reg signed [7:0] w_in_m_nc_d1_ch8;
  reg signed [7:0] w_in_m_nc_d1_ch9;
  reg signed [7:0] w_in_m_nc_d1_ch10;
  reg signed [7:0] w_in_m_nc_d1_ch11;
  reg signed [7:0] w_in_m_nc_d1_ch12;
  reg signed [7:0] w_in_m_nc_d1_ch13;
  reg signed [7:0] w_in_m_nc_d1_ch14;
  reg signed [7:0] w_in_m_nc_d1_ch15;
  reg signed [7:0] w_in_m_nc_d1_ch16;
  reg [2:0] state_nc_d1=0;
  reg [4:0] weight_select_d1;
  reg signed [0:31] buf0_ch1=0;
  reg signed [0:31] buf1_ch1=0;
  reg signed [0:31] buf2_ch1=0;
  reg signed [0:31] buf3_ch1=0;
  reg signed [0:31] buf4_ch1=0;
  reg signed [0:31] buf5_ch1=0;
  reg signed [0:31] buf6_ch1=0;
  reg signed [0:31] buf7_ch1=0;
  reg signed [0:31] buf8_ch1=0;
  reg signed [0:31] buf9_ch1=0;
  
  reg signed [0:31] buf0_ch2=0;
  reg signed [0:31] buf1_ch2=0;
  reg signed [0:31] buf2_ch2=0;
  reg signed [0:31] buf3_ch2=0;
  reg signed [0:31] buf4_ch2=0;
  reg signed [0:31] buf5_ch2=0;
  reg signed [0:31] buf6_ch2=0;
  reg signed [0:31] buf7_ch2=0;
  reg signed [0:31] buf8_ch2=0;
  reg signed [0:31] buf9_ch2=0;
  
  reg signed [0:31] buf0_ch3=0;
  reg signed [0:31] buf1_ch3=0;
  reg signed [0:31] buf2_ch3=0;
  reg signed [0:31] buf3_ch3=0;
  reg signed [0:31] buf4_ch3=0;
  reg signed [0:31] buf5_ch3=0;
  reg signed [0:31] buf6_ch3=0;
  reg signed [0:31] buf7_ch3=0;
  reg signed [0:31] buf8_ch3=0;
  reg signed [0:31] buf9_ch3=0;
  
  reg signed [0:31] buf0_ch4=0;
  reg signed [0:31] buf1_ch4=0;
  reg signed [0:31] buf2_ch4=0;
  reg signed [0:31] buf3_ch4=0;
  reg signed [0:31] buf4_ch4=0;
  reg signed [0:31] buf5_ch4=0;
  reg signed [0:31] buf6_ch4=0;
  reg signed [0:31] buf7_ch4=0;
  reg signed [0:31] buf8_ch4=0;
  reg signed [0:31] buf9_ch4=0;
  
  reg signed [0:31] buf0_ch5=0;
  reg signed [0:31] buf1_ch5=0;
  reg signed [0:31] buf2_ch5=0;
  reg signed [0:31] buf3_ch5=0;
  reg signed [0:31] buf4_ch5=0;
  reg signed [0:31] buf5_ch5=0;
  reg signed [0:31] buf6_ch5=0;
  reg signed [0:31] buf7_ch5=0;
  reg signed [0:31] buf8_ch5=0;
  reg signed [0:31] buf9_ch5=0;
  
  reg signed [0:31] buf0_ch6=0;
  reg signed [0:31] buf1_ch6=0;
  reg signed [0:31] buf2_ch6=0;
  reg signed [0:31] buf3_ch6=0;
  reg signed [0:31] buf4_ch6=0;
  reg signed [0:31] buf5_ch6=0;
  reg signed [0:31] buf6_ch6=0;
  reg signed [0:31] buf7_ch6=0;
  reg signed [0:31] buf8_ch6=0;
  reg signed [0:31] buf9_ch6=0;
  
  reg signed [0:31] buf0_ch7=0;
  reg signed [0:31] buf1_ch7=0;
  reg signed [0:31] buf2_ch7=0;
  reg signed [0:31] buf3_ch7=0;
  reg signed [0:31] buf4_ch7=0;
  reg signed [0:31] buf5_ch7=0;
  reg signed [0:31] buf6_ch7=0;
  reg signed [0:31] buf7_ch7=0;
  reg signed [0:31] buf8_ch7=0;
  reg signed [0:31] buf9_ch7=0;
  
  reg signed [0:31] buf0_ch8=0;
  reg signed [0:31] buf1_ch8=0;
  reg signed [0:31] buf2_ch8=0;
  reg signed [0:31] buf3_ch8=0;
  reg signed [0:31] buf4_ch8=0;
  reg signed [0:31] buf5_ch8=0;
  reg signed [0:31] buf6_ch8=0;
  reg signed [0:31] buf7_ch8=0;
  reg signed [0:31] buf8_ch8=0;
  reg signed [0:31] buf9_ch8=0;
  
  reg signed [0:31] buf0_ch9=0;
  reg signed [0:31] buf1_ch9=0;
  reg signed [0:31] buf2_ch9=0;
  reg signed [0:31] buf3_ch9=0;
  reg signed [0:31] buf4_ch9=0;
  reg signed [0:31] buf5_ch9=0;
  reg signed [0:31] buf6_ch9=0;
  reg signed [0:31] buf7_ch9=0;
  reg signed [0:31] buf8_ch9=0;
  reg signed [0:31] buf9_ch9=0;
  
  reg signed [0:31] buf0_ch10=0;
  reg signed [0:31] buf1_ch10=0;
  reg signed [0:31] buf2_ch10=0;
  reg signed [0:31] buf3_ch10=0;
  reg signed [0:31] buf4_ch10=0;
  reg signed [0:31] buf5_ch10=0;
  reg signed [0:31] buf6_ch10=0;
  reg signed [0:31] buf7_ch10=0;
  reg signed [0:31] buf8_ch10=0;
  reg signed [0:31] buf9_ch10=0;
  
  reg signed [0:31] buf0_ch11=0;
  reg signed [0:31] buf1_ch11=0;
  reg signed [0:31] buf2_ch11=0;
  reg signed [0:31] buf3_ch11=0;
  reg signed [0:31] buf4_ch11=0;
  reg signed [0:31] buf5_ch11=0;
  reg signed [0:31] buf6_ch11=0;
  reg signed [0:31] buf7_ch11=0;
  reg signed [0:31] buf8_ch11=0;
  reg signed [0:31] buf9_ch11=0;
  
  reg signed [0:31] buf0_ch12=0;
  reg signed [0:31] buf1_ch12=0;
  reg signed [0:31] buf2_ch12=0;
  reg signed [0:31] buf3_ch12=0;
  reg signed [0:31] buf4_ch12=0;
  reg signed [0:31] buf5_ch12=0;
  reg signed [0:31] buf6_ch12=0;
  reg signed [0:31] buf7_ch12=0;
  reg signed [0:31] buf8_ch12=0;
  reg signed [0:31] buf9_ch12=0;
  
  reg signed [0:31] buf0_ch13=0;
  reg signed [0:31] buf1_ch13=0;
  reg signed [0:31] buf2_ch13=0;
  reg signed [0:31] buf3_ch13=0;
  reg signed [0:31] buf4_ch13=0;
  reg signed [0:31] buf5_ch13=0;
  reg signed [0:31] buf6_ch13=0;
  reg signed [0:31] buf7_ch13=0;
  reg signed [0:31] buf8_ch13=0;
  reg signed [0:31] buf9_ch13=0;
  
  reg signed [0:31] buf0_ch14=0;
  reg signed [0:31] buf1_ch14=0;
  reg signed [0:31] buf2_ch14=0;
  reg signed [0:31] buf3_ch14=0;
  reg signed [0:31] buf4_ch14=0;
  reg signed [0:31] buf5_ch14=0;
  reg signed [0:31] buf6_ch14=0;
  reg signed [0:31] buf7_ch14=0;
  reg signed [0:31] buf8_ch14=0;
  reg signed [0:31] buf9_ch14=0;
  
  reg signed [0:31] buf0_ch15=0;
  reg signed [0:31] buf1_ch15=0;
  reg signed [0:31] buf2_ch15=0;
  reg signed [0:31] buf3_ch15=0;
  reg signed [0:31] buf4_ch15=0;
  reg signed [0:31] buf5_ch15=0;
  reg signed [0:31] buf6_ch15=0;
  reg signed [0:31] buf7_ch15=0;
  reg signed [0:31] buf8_ch15=0;
  reg signed [0:31] buf9_ch15=0;
  
  reg signed [0:31] buf0_ch16=0;
  reg signed [0:31] buf1_ch16=0;
  reg signed [0:31] buf2_ch16=0;
  reg signed [0:31] buf3_ch16=0;
  reg signed [0:31] buf4_ch16=0;
  reg signed [0:31] buf5_ch16=0;
  reg signed [0:31] buf6_ch16=0;
  reg signed [0:31] buf7_ch16=0;
  reg signed [0:31] buf8_ch16=0;
  reg signed [0:31] buf9_ch16=0;
  
  reg signed [0:31] p0_ch1;
  reg signed [0:31] p1_ch1;
  reg signed [0:31] p2_ch1;
  reg signed [0:31] p3_ch1;
  reg signed [0:31] p4_ch1;
  reg signed [0:31] p5_ch1;
  reg signed [0:31] p6_ch1;
  reg signed [0:31] p7_ch1;
  reg signed [0:31] p8_ch1;
  reg signed [0:31] p9_ch1;
  
  reg signed [0:31] p0_ch2;
  reg signed [0:31] p1_ch2;
  reg signed [0:31] p2_ch2;
  reg signed [0:31] p3_ch2;
  reg signed [0:31] p4_ch2;
  reg signed [0:31] p5_ch2;
  reg signed [0:31] p6_ch2;
  reg signed [0:31] p7_ch2;
  reg signed [0:31] p8_ch2;
  reg signed [0:31] p9_ch2;
  
  reg signed [0:31] p0_ch3;
  reg signed [0:31] p1_ch3;
  reg signed [0:31] p2_ch3;
  reg signed [0:31] p3_ch3;
  reg signed [0:31] p4_ch3;
  reg signed [0:31] p5_ch3;
  reg signed [0:31] p6_ch3;
  reg signed [0:31] p7_ch3;
  reg signed [0:31] p8_ch3;
  reg signed [0:31] p9_ch3;
  
  reg signed [0:31] p0_ch4;
  reg signed [0:31] p1_ch4;
  reg signed [0:31] p2_ch4;
  reg signed [0:31] p3_ch4;
  reg signed [0:31] p4_ch4;
  reg signed [0:31] p5_ch4;
  reg signed [0:31] p6_ch4;
  reg signed [0:31] p7_ch4;
  reg signed [0:31] p8_ch4;
  reg signed [0:31] p9_ch4;
  
  reg signed [0:31] p0_ch5;
  reg signed [0:31] p1_ch5;
  reg signed [0:31] p2_ch5;
  reg signed [0:31] p3_ch5;
  reg signed [0:31] p4_ch5;
  reg signed [0:31] p5_ch5;
  reg signed [0:31] p6_ch5;
  reg signed [0:31] p7_ch5;
  reg signed [0:31] p8_ch5;
  reg signed [0:31] p9_ch5;
  
  reg signed [0:31] p0_ch6;
  reg signed [0:31] p1_ch6;
  reg signed [0:31] p2_ch6;
  reg signed [0:31] p3_ch6;
  reg signed [0:31] p4_ch6;
  reg signed [0:31] p5_ch6;
  reg signed [0:31] p6_ch6;
  reg signed [0:31] p7_ch6;
  reg signed [0:31] p8_ch6;
  reg signed [0:31] p9_ch6;
  
  reg signed [0:31] p0_ch7;
  reg signed [0:31] p1_ch7;
  reg signed [0:31] p2_ch7;
  reg signed [0:31] p3_ch7;
  reg signed [0:31] p4_ch7;
  reg signed [0:31] p5_ch7;
  reg signed [0:31] p6_ch7;
  reg signed [0:31] p7_ch7;
  reg signed [0:31] p8_ch7;
  reg signed [0:31] p9_ch7;
  
  reg signed [0:31] p0_ch8;
  reg signed [0:31] p1_ch8;
  reg signed [0:31] p2_ch8;
  reg signed [0:31] p3_ch8;
  reg signed [0:31] p4_ch8;
  reg signed [0:31] p5_ch8;
  reg signed [0:31] p6_ch8;
  reg signed [0:31] p7_ch8;
  reg signed [0:31] p8_ch8;
  reg signed [0:31] p9_ch8;
  
  reg signed [0:31] p0_ch9;
  reg signed [0:31] p1_ch9;
  reg signed [0:31] p2_ch9;
  reg signed [0:31] p3_ch9;
  reg signed [0:31] p4_ch9;
  reg signed [0:31] p5_ch9;
  reg signed [0:31] p6_ch9;
  reg signed [0:31] p7_ch9;
  reg signed [0:31] p8_ch9;
  reg signed [0:31] p9_ch9;
  
  reg signed [0:31] p0_ch10;
  reg signed [0:31] p1_ch10;
  reg signed [0:31] p2_ch10;
  reg signed [0:31] p3_ch10;
  reg signed [0:31] p4_ch10;
  reg signed [0:31] p5_ch10;
  reg signed [0:31] p6_ch10;
  reg signed [0:31] p7_ch10;
  reg signed [0:31] p8_ch10;
  reg signed [0:31] p9_ch10;
  
  reg signed [0:31] p0_ch11;
  reg signed [0:31] p1_ch11;
  reg signed [0:31] p2_ch11;
  reg signed [0:31] p3_ch11;
  reg signed [0:31] p4_ch11;
  reg signed [0:31] p5_ch11;
  reg signed [0:31] p6_ch11;
  reg signed [0:31] p7_ch11;
  reg signed [0:31] p8_ch11;
  reg signed [0:31] p9_ch11;
  
  reg signed [0:31] p0_ch12;
  reg signed [0:31] p1_ch12;
  reg signed [0:31] p2_ch12;
  reg signed [0:31] p3_ch12;
  reg signed [0:31] p4_ch12;
  reg signed [0:31] p5_ch12;
  reg signed [0:31] p6_ch12;
  reg signed [0:31] p7_ch12;
  reg signed [0:31] p8_ch12;
  reg signed [0:31] p9_ch12;
  
  reg signed [0:31] p0_ch13;
  reg signed [0:31] p1_ch13;
  reg signed [0:31] p2_ch13;
  reg signed [0:31] p3_ch13;
  reg signed [0:31] p4_ch13;
  reg signed [0:31] p5_ch13;
  reg signed [0:31] p6_ch13;
  reg signed [0:31] p7_ch13;
  reg signed [0:31] p8_ch13;
  reg signed [0:31] p9_ch13;
  
  reg signed [0:31] p0_ch14;
  reg signed [0:31] p1_ch14;
  reg signed [0:31] p2_ch14;
  reg signed [0:31] p3_ch14;
  reg signed [0:31] p4_ch14;
  reg signed [0:31] p5_ch14;
  reg signed [0:31] p6_ch14;
  reg signed [0:31] p7_ch14;
  reg signed [0:31] p8_ch14;
  reg signed [0:31] p9_ch14;
  
  reg signed [0:31] p0_ch15;
  reg signed [0:31] p1_ch15;
  reg signed [0:31] p2_ch15;
  reg signed [0:31] p3_ch15;
  reg signed [0:31] p4_ch15;
  reg signed [0:31] p5_ch15;
  reg signed [0:31] p6_ch15;
  reg signed [0:31] p7_ch15;
  reg signed [0:31] p8_ch15;
  reg signed [0:31] p9_ch15;
  
  reg signed [0:31] p0_ch16;
  reg signed [0:31] p1_ch16;
  reg signed [0:31] p2_ch16;
  reg signed [0:31] p3_ch16;
  reg signed [0:31] p4_ch16;
  reg signed [0:31] p5_ch16;
  reg signed [0:31] p6_ch16;
  reg signed [0:31] p7_ch16;
  reg signed [0:31] p8_ch16;
  reg signed [0:31] p9_ch16;
  
////////////////////////////////////////////////// softmax ///////////////////////////////////////////////////////////////
  reg signed [0:31] values [0:9];
  reg [0:3]temp=4'h0;
  reg signed [0:31]max=32'h80000001;
  integer i;
  
  assign start_addr_conv_c1 = window_select_c1+(window_select_c1/26)*2;
  assign start_addr_conv_c2 = window_select_c2+(window_select_c2/11)*2;
///////////////////////////////// BRAM for parameter storage D1 layer/////////////////////////////////////
  bram #(ADDR_WIDTH,DATA_WIDTH,DEPTH) BRAM_D1(
       .clk(clk),
       .addr(addr_d1),
       .wr_en(wr_en_d1),
       .data_in(din_d1),
       .data_out(dout_d1));
/////////////////////////////// BRAM for parameter storage C1 & C1 port0-C1, port1-C2//////////////////////
  dualport_bram_param #(ADDR_WIDTH,DATA_WIDTH,DEPTH) BRAM(
       .clk(clk),
       .addr_0(addr),
       .addr_1(addr_c2),
       .wr_en_0(wr_en),
       .wr_en_1(wr_en_c2),
       .oe_0(oe),
       .oe_1(oe_c2),
       .data_in_0(din),
       .data_in_1(din_c2),
       .data_out_0(dout),
       .data_out_1(dout_c2));
///////////////////////////////////// BRAM for C1 channel 1 //////////////////////////////////////
  dualport_bram_c1 #(ADDR_WIDTH_BUF_C1,DATA_WIDTH_BUF,DEPTH_BUF_C1) D_BRAM_C1_CH1(
       .clk(clk),
       .addr_0(addr_buf_0_c1_ch1),
       .addr_1(addr_buf_1_c1_ch1),
       .wr_en_0(wr_en_buf_0_c1_ch1),
       .wr_en_1(wr_en_buf_1_c1_ch1),
       .oe_0(oe_buf_0_c1_ch1),
       .oe_1(oe_buf_1_c1_ch1),
       .data_in_0(din_buf_0_c1_ch1),
       .data_in_1(din_buf_1_c1_ch1),
       .data_out_0(dout_buf_0_c1_ch1),
       .data_out_1(dout_buf_1_c1_ch1));
///////////////////////////////////// BRAM for C1 channel 2 //////////////////////////////////////
  dualport_bram_c1 #(ADDR_WIDTH_BUF_C1,DATA_WIDTH_BUF,DEPTH_BUF_C1) D_BRAM_C1_CH2(
       .clk(clk),
       .addr_0(addr_buf_0_c1_ch2),
       .addr_1(addr_buf_1_c1_ch2),
       .wr_en_0(wr_en_buf_0_c1_ch2),
       .wr_en_1(wr_en_buf_1_c1_ch2),
       .oe_0(oe_buf_0_c1_ch2),
       .oe_1(oe_buf_1_c1_ch2),
       .data_in_0(din_buf_0_c1_ch2),
       .data_in_1(din_buf_1_c1_ch2),
       .data_out_0(dout_buf_0_c1_ch2),
       .data_out_1(dout_buf_1_c1_ch2));
///////////////////////////////////// BRAM for C1 channel 3 //////////////////////////////////////
  dualport_bram_c1 #(ADDR_WIDTH_BUF_C1,DATA_WIDTH_BUF,DEPTH_BUF_C1) D_BRAM_C1_CH3(
       .clk(clk),
       .addr_0(addr_buf_0_c1_ch3),
       .addr_1(addr_buf_1_c1_ch3),
       .wr_en_0(wr_en_buf_0_c1_ch3),
       .wr_en_1(wr_en_buf_1_c1_ch3),
       .oe_0(oe_buf_0_c1_ch3),
       .oe_1(oe_buf_1_c1_ch3),
       .data_in_0(din_buf_0_c1_ch3),
       .data_in_1(din_buf_1_c1_ch3),
       .data_out_0(dout_buf_0_c1_ch3),
       .data_out_1(dout_buf_1_c1_ch3));
///////////////////////////////////// BRAM for C1 channel 4 //////////////////////////////////////
  dualport_bram_c1 #(ADDR_WIDTH_BUF_C1,DATA_WIDTH_BUF,DEPTH_BUF_C1) D_BRAM_C1_CH4(
       .clk(clk),
       .addr_0(addr_buf_0_c1_ch4),
       .addr_1(addr_buf_1_c1_ch4),
       .wr_en_0(wr_en_buf_0_c1_ch4),
       .wr_en_1(wr_en_buf_1_c1_ch4),
       .oe_0(oe_buf_0_c1_ch4),
       .oe_1(oe_buf_1_c1_ch4),
       .data_in_0(din_buf_0_c1_ch4),
       .data_in_1(din_buf_1_c1_ch4),
       .data_out_0(dout_buf_0_c1_ch4),
       .data_out_1(dout_buf_1_c1_ch4));
///////////////////////////////////// BRAM for C1 channel 5 //////////////////////////////////////
  dualport_bram_c1 #(ADDR_WIDTH_BUF_C1,DATA_WIDTH_BUF,DEPTH_BUF_C1) D_BRAM_C1_CH5(
       .clk(clk),
       .addr_0(addr_buf_0_c1_ch5),
       .addr_1(addr_buf_1_c1_ch5),
       .wr_en_0(wr_en_buf_0_c1_ch5),
       .wr_en_1(wr_en_buf_1_c1_ch5),
       .oe_0(oe_buf_0_c1_ch5),
       .oe_1(oe_buf_1_c1_ch5),
       .data_in_0(din_buf_0_c1_ch5),
       .data_in_1(din_buf_1_c1_ch5),
       .data_out_0(dout_buf_0_c1_ch5),
       .data_out_1(dout_buf_1_c1_ch5));
///////////////////////////////////// BRAM for C1 channel 6 //////////////////////////////////////
  dualport_bram_c1 #(ADDR_WIDTH_BUF_C1,DATA_WIDTH_BUF,DEPTH_BUF_C1) D_BRAM_C1_CH6(
       .clk(clk),
       .addr_0(addr_buf_0_c1_ch6),
       .addr_1(addr_buf_1_c1_ch6),
       .wr_en_0(wr_en_buf_0_c1_ch6),
       .wr_en_1(wr_en_buf_1_c1_ch6),
       .oe_0(oe_buf_0_c1_ch6),
       .oe_1(oe_buf_1_c1_ch6),
       .data_in_0(din_buf_0_c1_ch6),
       .data_in_1(din_buf_1_c1_ch6),
       .data_out_0(dout_buf_0_c1_ch6),
       .data_out_1(dout_buf_1_c1_ch6));
///////////////////////////////////// BRAM for M1 channel 1 //////////////////////////////////////
  dualport_bram_m1 #(ADDR_WIDTH_BUF_M1,DATA_WIDTH_BUF,DEPTH_BUF_M1) D_BRAM_M1_CH1(
       .clk(clk),
       .addr_0(addr_buf_0_m1_ch1),
       .addr_1(addr_buf_1_m1_ch1),
       .wr_en_0(wr_en_buf_0_m1_ch1),
       .wr_en_1(wr_en_buf_1_m1_ch1),
       .oe_0(oe_buf_0_m1_ch1),
       .oe_1(oe_buf_1_m1_ch1),
       .data_in_0(din_buf_0_m1_ch1),
       .data_in_1(din_buf_1_m1_ch1),
       .data_out_0(dout_buf_0_m1_ch1),
       .data_out_1(dout_buf_1_m1_ch1));
///////////////////////////////////// BRAM for M1 channel 2 //////////////////////////////////////
  dualport_bram_m1 #(ADDR_WIDTH_BUF_M1,DATA_WIDTH_BUF,DEPTH_BUF_M1) D_BRAM_M1_CH2(
       .clk(clk),
       .addr_0(addr_buf_0_m1_ch2),
       .addr_1(addr_buf_1_m1_ch2),
       .wr_en_0(wr_en_buf_0_m1_ch2),
       .wr_en_1(wr_en_buf_1_m1_ch2),
       .oe_0(oe_buf_0_m1_ch2),
       .oe_1(oe_buf_1_m1_ch2),
       .data_in_0(din_buf_0_m1_ch2),
       .data_in_1(din_buf_1_m1_ch2),
       .data_out_0(dout_buf_0_m1_ch2),
       .data_out_1(dout_buf_1_m1_ch2));
///////////////////////////////////// BRAM for M1 channel 3 //////////////////////////////////////
  dualport_bram_m1 #(ADDR_WIDTH_BUF_M1,DATA_WIDTH_BUF,DEPTH_BUF_M1) D_BRAM_M1_CH3(
       .clk(clk),
       .addr_0(addr_buf_0_m1_ch3),
       .addr_1(addr_buf_1_m1_ch3),
       .wr_en_0(wr_en_buf_0_m1_ch3),
       .wr_en_1(wr_en_buf_1_m1_ch3),
       .oe_0(oe_buf_0_m1_ch3),
       .oe_1(oe_buf_1_m1_ch3),
       .data_in_0(din_buf_0_m1_ch3),
       .data_in_1(din_buf_1_m1_ch3),
       .data_out_0(dout_buf_0_m1_ch3),
       .data_out_1(dout_buf_1_m1_ch3));
///////////////////////////////////// BRAM for M1 channel 4 //////////////////////////////////////
  dualport_bram_m1 #(ADDR_WIDTH_BUF_M1,DATA_WIDTH_BUF,DEPTH_BUF_M1) D_BRAM_M1_CH4(
       .clk(clk),
       .addr_0(addr_buf_0_m1_ch4),
       .addr_1(addr_buf_1_m1_ch4),
       .wr_en_0(wr_en_buf_0_m1_ch4),
       .wr_en_1(wr_en_buf_1_m1_ch4),
       .oe_0(oe_buf_0_m1_ch4),
       .oe_1(oe_buf_1_m1_ch4),
       .data_in_0(din_buf_0_m1_ch4),
       .data_in_1(din_buf_1_m1_ch4),
       .data_out_0(dout_buf_0_m1_ch4),
       .data_out_1(dout_buf_1_m1_ch4));
///////////////////////////////////// BRAM for M1 channel 5 //////////////////////////////////////
  dualport_bram_m1 #(ADDR_WIDTH_BUF_M1,DATA_WIDTH_BUF,DEPTH_BUF_M1) D_BRAM_M1_CH5(
       .clk(clk),
       .addr_0(addr_buf_0_m1_ch5),
       .addr_1(addr_buf_1_m1_ch5),
       .wr_en_0(wr_en_buf_0_m1_ch5),
       .wr_en_1(wr_en_buf_1_m1_ch5),
       .oe_0(oe_buf_0_m1_ch5),
       .oe_1(oe_buf_1_m1_ch5),
       .data_in_0(din_buf_0_m1_ch5),
       .data_in_1(din_buf_1_m1_ch5),
       .data_out_0(dout_buf_0_m1_ch5),
       .data_out_1(dout_buf_1_m1_ch5));
///////////////////////////////////// BRAM for M1 channel 6 //////////////////////////////////////
  dualport_bram_m1 #(ADDR_WIDTH_BUF_M1,DATA_WIDTH_BUF,DEPTH_BUF_M1) D_BRAM_M1_CH6(
       .clk(clk),
       .addr_0(addr_buf_0_m1_ch6),
       .addr_1(addr_buf_1_m1_ch6),
       .wr_en_0(wr_en_buf_0_m1_ch6),
       .wr_en_1(wr_en_buf_1_m1_ch6),
       .oe_0(oe_buf_0_m1_ch6),
       .oe_1(oe_buf_1_m1_ch6),
       .data_in_0(din_buf_0_m1_ch6),
       .data_in_1(din_buf_1_m1_ch6),
       .data_out_0(dout_buf_0_m1_ch6),
       .data_out_1(dout_buf_1_m1_ch6));
///////////////////////////////////// BRAM for C2 channel 1 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH1(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch1),
       .addr_1(addr_buf_1_c2_ch1),
       .wr_en_0(wr_en_buf_0_c2_ch1),
       .wr_en_1(wr_en_buf_1_c2_ch1),
       .oe_0(oe_buf_0_c2_ch1),
       .oe_1(oe_buf_1_c2_ch1),
       .data_in_0(din_buf_0_c2_ch1),
       .data_in_1(din_buf_1_c2_ch1),
       .data_out_0(dout_buf_0_c2_ch1),
       .data_out_1(dout_buf_1_c2_ch1));
///////////////////////////////////// BRAM for C2 channel 2 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH2(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch2),
       .addr_1(addr_buf_1_c2_ch2),
       .wr_en_0(wr_en_buf_0_c2_ch2),
       .wr_en_1(wr_en_buf_1_c2_ch2),
       .oe_0(oe_buf_0_c2_ch2),
       .oe_1(oe_buf_1_c2_ch2),
       .data_in_0(din_buf_0_c2_ch2),
       .data_in_1(din_buf_1_c2_ch2),
       .data_out_0(dout_buf_0_c2_ch2),
       .data_out_1(dout_buf_1_c2_ch2));
///////////////////////////////////// BRAM for C2 channel 3 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH3(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch3),
       .addr_1(addr_buf_1_c2_ch3),
       .wr_en_0(wr_en_buf_0_c2_ch3),
       .wr_en_1(wr_en_buf_1_c2_ch3),
       .oe_0(oe_buf_0_c2_ch3),
       .oe_1(oe_buf_1_c2_ch3),
       .data_in_0(din_buf_0_c2_ch3),
       .data_in_1(din_buf_1_c2_ch3),
       .data_out_0(dout_buf_0_c2_ch3),
       .data_out_1(dout_buf_1_c2_ch3));
///////////////////////////////////// BRAM for C2 channel 4 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH4(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch4),
       .addr_1(addr_buf_1_c2_ch4),
       .wr_en_0(wr_en_buf_0_c2_ch4),
       .wr_en_1(wr_en_buf_1_c2_ch4),
       .oe_0(oe_buf_0_c2_ch4),
       .oe_1(oe_buf_1_c2_ch4),
       .data_in_0(din_buf_0_c2_ch4),
       .data_in_1(din_buf_1_c2_ch4),
       .data_out_0(dout_buf_0_c2_ch4),
       .data_out_1(dout_buf_1_c2_ch4));
///////////////////////////////////// BRAM for C2 channel 5 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH5(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch5),
       .addr_1(addr_buf_1_c2_ch5),
       .wr_en_0(wr_en_buf_0_c2_ch5),
       .wr_en_1(wr_en_buf_1_c2_ch5),
       .oe_0(oe_buf_0_c2_ch5),
       .oe_1(oe_buf_1_c2_ch5),
       .data_in_0(din_buf_0_c2_ch5),
       .data_in_1(din_buf_1_c2_ch5),
       .data_out_0(dout_buf_0_c2_ch5),
       .data_out_1(dout_buf_1_c2_ch5));
///////////////////////////////////// BRAM for C2 channel 6 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH6(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch6),
       .addr_1(addr_buf_1_c2_ch6),
       .wr_en_0(wr_en_buf_0_c2_ch6),
       .wr_en_1(wr_en_buf_1_c2_ch6),
       .oe_0(oe_buf_0_c2_ch6),
       .oe_1(oe_buf_1_c2_ch6),
       .data_in_0(din_buf_0_c2_ch6),
       .data_in_1(din_buf_1_c2_ch6),
       .data_out_0(dout_buf_0_c2_ch6),
       .data_out_1(dout_buf_1_c2_ch6));
///////////////////////////////////// BRAM for C2 channel 7 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH7(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch7),
       .addr_1(addr_buf_1_c2_ch7),
       .wr_en_0(wr_en_buf_0_c2_ch7),
       .wr_en_1(wr_en_buf_1_c2_ch7),
       .oe_0(oe_buf_0_c2_ch7),
       .oe_1(oe_buf_1_c2_ch7),
       .data_in_0(din_buf_0_c2_ch7),
       .data_in_1(din_buf_1_c2_ch7),
       .data_out_0(dout_buf_0_c2_ch7),
       .data_out_1(dout_buf_1_c2_ch7));
///////////////////////////////////// BRAM for C2 channel 8 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH8(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch8),
       .addr_1(addr_buf_1_c2_ch8),
       .wr_en_0(wr_en_buf_0_c2_ch8),
       .wr_en_1(wr_en_buf_1_c2_ch8),
       .oe_0(oe_buf_0_c2_ch8),
       .oe_1(oe_buf_1_c2_ch8),
       .data_in_0(din_buf_0_c2_ch8),
       .data_in_1(din_buf_1_c2_ch8),
       .data_out_0(dout_buf_0_c2_ch8),
       .data_out_1(dout_buf_1_c2_ch8));
///////////////////////////////////// BRAM for C2 channel 9 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH9(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch9),
       .addr_1(addr_buf_1_c2_ch9),
       .wr_en_0(wr_en_buf_0_c2_ch9),
       .wr_en_1(wr_en_buf_1_c2_ch9),
       .oe_0(oe_buf_0_c2_ch9),
       .oe_1(oe_buf_1_c2_ch9),
       .data_in_0(din_buf_0_c2_ch9),
       .data_in_1(din_buf_1_c2_ch9),
       .data_out_0(dout_buf_0_c2_ch9),
       .data_out_1(dout_buf_1_c2_ch9));
///////////////////////////////////// BRAM for C2 channel 10 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH10(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch10),
       .addr_1(addr_buf_1_c2_ch10),
       .wr_en_0(wr_en_buf_0_c2_ch10),
       .wr_en_1(wr_en_buf_1_c2_ch10),
       .oe_0(oe_buf_0_c2_ch10),
       .oe_1(oe_buf_1_c2_ch10),
       .data_in_0(din_buf_0_c2_ch10),
       .data_in_1(din_buf_1_c2_ch10),
       .data_out_0(dout_buf_0_c2_ch10),
       .data_out_1(dout_buf_1_c2_ch10));
///////////////////////////////////// BRAM for C2 channel 11 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH11(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch11),
       .addr_1(addr_buf_1_c2_ch11),
       .wr_en_0(wr_en_buf_0_c2_ch11),
       .wr_en_1(wr_en_buf_1_c2_ch11),
       .oe_0(oe_buf_0_c2_ch11),
       .oe_1(oe_buf_1_c2_ch11),
       .data_in_0(din_buf_0_c2_ch11),
       .data_in_1(din_buf_1_c2_ch11),
       .data_out_0(dout_buf_0_c2_ch11),
       .data_out_1(dout_buf_1_c2_ch11));
///////////////////////////////////// BRAM for C2 channel 12 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH12(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch12),
       .addr_1(addr_buf_1_c2_ch12),
       .wr_en_0(wr_en_buf_0_c2_ch12),
       .wr_en_1(wr_en_buf_1_c2_ch12),
       .oe_0(oe_buf_0_c2_ch12),
       .oe_1(oe_buf_1_c2_ch12),
       .data_in_0(din_buf_0_c2_ch12),
       .data_in_1(din_buf_1_c2_ch12),
       .data_out_0(dout_buf_0_c2_ch12),
       .data_out_1(dout_buf_1_c2_ch12));
///////////////////////////////////// BRAM for C2 channel 13 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH13(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch13),
       .addr_1(addr_buf_1_c2_ch13),
       .wr_en_0(wr_en_buf_0_c2_ch13),
       .wr_en_1(wr_en_buf_1_c2_ch13),
       .oe_0(oe_buf_0_c2_ch13),
       .oe_1(oe_buf_1_c2_ch13),
       .data_in_0(din_buf_0_c2_ch13),
       .data_in_1(din_buf_1_c2_ch13),
       .data_out_0(dout_buf_0_c2_ch13),
       .data_out_1(dout_buf_1_c2_ch13));
///////////////////////////////////// BRAM for C2 channel 14 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH14(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch14),
       .addr_1(addr_buf_1_c2_ch14),
       .wr_en_0(wr_en_buf_0_c2_ch14),
       .wr_en_1(wr_en_buf_1_c2_ch14),
       .oe_0(oe_buf_0_c2_ch14),
       .oe_1(oe_buf_1_c2_ch14),
       .data_in_0(din_buf_0_c2_ch14),
       .data_in_1(din_buf_1_c2_ch14),
       .data_out_0(dout_buf_0_c2_ch14),
       .data_out_1(dout_buf_1_c2_ch14));
///////////////////////////////////// BRAM for C2 channel 15 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH15(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch15),
       .addr_1(addr_buf_1_c2_ch15),
       .wr_en_0(wr_en_buf_0_c2_ch15),
       .wr_en_1(wr_en_buf_1_c2_ch15),
       .oe_0(oe_buf_0_c2_ch15),
       .oe_1(oe_buf_1_c2_ch15),
       .data_in_0(din_buf_0_c2_ch15),
       .data_in_1(din_buf_1_c2_ch15),
       .data_out_0(dout_buf_0_c2_ch15),
       .data_out_1(dout_buf_1_c2_ch15));
///////////////////////////////////// BRAM for C2 channel 16 //////////////////////////////////////
  dualport_bram_c2 #(ADDR_WIDTH_BUF_C2,DATA_WIDTH_BUF,DEPTH_BUF_C2) D_BRAM_C2_CH16(
       .clk(clk),
       .addr_0(addr_buf_0_c2_ch16),
       .addr_1(addr_buf_1_c2_ch16),
       .wr_en_0(wr_en_buf_0_c2_ch16),
       .wr_en_1(wr_en_buf_1_c2_ch16),
       .oe_0(oe_buf_0_c2_ch16),
       .oe_1(oe_buf_1_c2_ch16),
       .data_in_0(din_buf_0_c2_ch16),
       .data_in_1(din_buf_1_c2_ch16),
       .data_out_0(dout_buf_0_c2_ch16),
       .data_out_1(dout_buf_1_c2_ch16));
///////////////////////////////////// BRAM for M2 channel 1 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH1(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch1),
       .addr_1(addr_buf_1_m2_ch1),
       .wr_en_0(wr_en_buf_0_m2_ch1),
       .wr_en_1(wr_en_buf_1_m2_ch1),
       .oe_0(oe_buf_0_m2_ch1),
       .oe_1(oe_buf_1_m2_ch1),
       .data_in_0(din_buf_0_m2_ch1),
       .data_in_1(din_buf_1_m2_ch1),
       .data_out_0(dout_buf_0_m2_ch1),
       .data_out_1(dout_buf_1_m2_ch1));
///////////////////////////////////// BRAM for M2 channel 2 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH2(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch2),
       .addr_1(addr_buf_1_m2_ch2),
       .wr_en_0(wr_en_buf_0_m2_ch2),
       .wr_en_1(wr_en_buf_1_m2_ch2),
       .oe_0(oe_buf_0_m2_ch2),
       .oe_1(oe_buf_1_m2_ch2),
       .data_in_0(din_buf_0_m2_ch2),
       .data_in_1(din_buf_1_m2_ch2),
       .data_out_0(dout_buf_0_m2_ch2),
       .data_out_1(dout_buf_1_m2_ch2));
///////////////////////////////////// BRAM for M2 channel 3 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH3(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch3),
       .addr_1(addr_buf_1_m2_ch3),
       .wr_en_0(wr_en_buf_0_m2_ch3),
       .wr_en_1(wr_en_buf_1_m2_ch3),
       .oe_0(oe_buf_0_m2_ch3),
       .oe_1(oe_buf_1_m2_ch3),
       .data_in_0(din_buf_0_m2_ch3),
       .data_in_1(din_buf_1_m2_ch3),
       .data_out_0(dout_buf_0_m2_ch3),
       .data_out_1(dout_buf_1_m2_ch3));
///////////////////////////////////// BRAM for M2 channel 4 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH4(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch4),
       .addr_1(addr_buf_1_m2_ch4),
       .wr_en_0(wr_en_buf_0_m2_ch4),
       .wr_en_1(wr_en_buf_1_m2_ch4),
       .oe_0(oe_buf_0_m2_ch4),
       .oe_1(oe_buf_1_m2_ch4),
       .data_in_0(din_buf_0_m2_ch4),
       .data_in_1(din_buf_1_m2_ch4),
       .data_out_0(dout_buf_0_m2_ch4),
       .data_out_1(dout_buf_1_m2_ch4));
///////////////////////////////////// BRAM for M2 channel 5 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH5(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch5),
       .addr_1(addr_buf_1_m2_ch5),
       .wr_en_0(wr_en_buf_0_m2_ch5),
       .wr_en_1(wr_en_buf_1_m2_ch5),
       .oe_0(oe_buf_0_m2_ch5),
       .oe_1(oe_buf_1_m2_ch5),
       .data_in_0(din_buf_0_m2_ch5),
       .data_in_1(din_buf_1_m2_ch5),
       .data_out_0(dout_buf_0_m2_ch5),
       .data_out_1(dout_buf_1_m2_ch5));
///////////////////////////////////// BRAM for M2 channel 6 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH6(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch6),
       .addr_1(addr_buf_1_m2_ch6),
       .wr_en_0(wr_en_buf_0_m2_ch6),
       .wr_en_1(wr_en_buf_1_m2_ch6),
       .oe_0(oe_buf_0_m2_ch6),
       .oe_1(oe_buf_1_m2_ch6),
       .data_in_0(din_buf_0_m2_ch6),
       .data_in_1(din_buf_1_m2_ch6),
       .data_out_0(dout_buf_0_m2_ch6),
       .data_out_1(dout_buf_1_m2_ch6));
///////////////////////////////////// BRAM for M2 channel 7 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH7(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch7),
       .addr_1(addr_buf_1_m2_ch7),
       .wr_en_0(wr_en_buf_0_m2_ch7),
       .wr_en_1(wr_en_buf_1_m2_ch7),
       .oe_0(oe_buf_0_m2_ch7),
       .oe_1(oe_buf_1_m2_ch7),
       .data_in_0(din_buf_0_m2_ch7),
       .data_in_1(din_buf_1_m2_ch7),
       .data_out_0(dout_buf_0_m2_ch7),
       .data_out_1(dout_buf_1_m2_ch7));
///////////////////////////////////// BRAM for M2 channel 8 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH8(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch8),
       .addr_1(addr_buf_1_m2_ch8),
       .wr_en_0(wr_en_buf_0_m2_ch8),
       .wr_en_1(wr_en_buf_1_m2_ch8),
       .oe_0(oe_buf_0_m2_ch8),
       .oe_1(oe_buf_1_m2_ch8),
       .data_in_0(din_buf_0_m2_ch8),
       .data_in_1(din_buf_1_m2_ch8),
       .data_out_0(dout_buf_0_m2_ch8),
       .data_out_1(dout_buf_1_m2_ch8));
///////////////////////////////////// BRAM for M2 channel 9 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH9(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch9),
       .addr_1(addr_buf_1_m2_ch9),
       .wr_en_0(wr_en_buf_0_m2_ch9),
       .wr_en_1(wr_en_buf_1_m2_ch9),
       .oe_0(oe_buf_0_m2_ch9),
       .oe_1(oe_buf_1_m2_ch9),
       .data_in_0(din_buf_0_m2_ch9),
       .data_in_1(din_buf_1_m2_ch9),
       .data_out_0(dout_buf_0_m2_ch9),
       .data_out_1(dout_buf_1_m2_ch9));
///////////////////////////////////// BRAM for M2 channel 10 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH10(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch10),
       .addr_1(addr_buf_1_m2_ch10),
       .wr_en_0(wr_en_buf_0_m2_ch10),
       .wr_en_1(wr_en_buf_1_m2_ch10),
       .oe_0(oe_buf_0_m2_ch10),
       .oe_1(oe_buf_1_m2_ch10),
       .data_in_0(din_buf_0_m2_ch10),
       .data_in_1(din_buf_1_m2_ch10),
       .data_out_0(dout_buf_0_m2_ch10),
       .data_out_1(dout_buf_1_m2_ch10));
///////////////////////////////////// BRAM for M2 channel 11 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH11(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch11),
       .addr_1(addr_buf_1_m2_ch11),
       .wr_en_0(wr_en_buf_0_m2_ch11),
       .wr_en_1(wr_en_buf_1_m2_ch11),
       .oe_0(oe_buf_0_m2_ch11),
       .oe_1(oe_buf_1_m2_ch11),
       .data_in_0(din_buf_0_m2_ch11),
       .data_in_1(din_buf_1_m2_ch11),
       .data_out_0(dout_buf_0_m2_ch11),
       .data_out_1(dout_buf_1_m2_ch11));
///////////////////////////////////// BRAM for M2 channel 12 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH12(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch12),
       .addr_1(addr_buf_1_m2_ch12),
       .wr_en_0(wr_en_buf_0_m2_ch12),
       .wr_en_1(wr_en_buf_1_m2_ch12),
       .oe_0(oe_buf_0_m2_ch12),
       .oe_1(oe_buf_1_m2_ch12),
       .data_in_0(din_buf_0_m2_ch12),
       .data_in_1(din_buf_1_m2_ch12),
       .data_out_0(dout_buf_0_m2_ch12),
       .data_out_1(dout_buf_1_m2_ch12));
///////////////////////////////////// BRAM for M2 channel 13 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH13(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch13),
       .addr_1(addr_buf_1_m2_ch13),
       .wr_en_0(wr_en_buf_0_m2_ch13),
       .wr_en_1(wr_en_buf_1_m2_ch13),
       .oe_0(oe_buf_0_m2_ch13),
       .oe_1(oe_buf_1_m2_ch13),
       .data_in_0(din_buf_0_m2_ch13),
       .data_in_1(din_buf_1_m2_ch13),
       .data_out_0(dout_buf_0_m2_ch13),
       .data_out_1(dout_buf_1_m2_ch13));
///////////////////////////////////// BRAM for M2 channel 14 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH14(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch14),
       .addr_1(addr_buf_1_m2_ch14),
       .wr_en_0(wr_en_buf_0_m2_ch14),
       .wr_en_1(wr_en_buf_1_m2_ch14),
       .oe_0(oe_buf_0_m2_ch14),
       .oe_1(oe_buf_1_m2_ch14),
       .data_in_0(din_buf_0_m2_ch14),
       .data_in_1(din_buf_1_m2_ch14),
       .data_out_0(dout_buf_0_m2_ch14),
       .data_out_1(dout_buf_1_m2_ch14));
///////////////////////////////////// BRAM for M2 channel 15 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH15(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch15),
       .addr_1(addr_buf_1_m2_ch15),
       .wr_en_0(wr_en_buf_0_m2_ch15),
       .wr_en_1(wr_en_buf_1_m2_ch15),
       .oe_0(oe_buf_0_m2_ch15),
       .oe_1(oe_buf_1_m2_ch15),
       .data_in_0(din_buf_0_m2_ch15),
       .data_in_1(din_buf_1_m2_ch15),
       .data_out_0(dout_buf_0_m2_ch15),
       .data_out_1(dout_buf_1_m2_ch15));
///////////////////////////////////// BRAM for M2 channel 16 //////////////////////////////////////
  dualport_bram_m2 #(ADDR_WIDTH_BUF_M2,DATA_WIDTH_BUF,DEPTH_BUF_M2) D_BRAM_M2_CH16(
       .clk(clk),
       .addr_0(addr_buf_0_m2_ch16),
       .addr_1(addr_buf_1_m2_ch16),
       .wr_en_0(wr_en_buf_0_m2_ch16),
       .wr_en_1(wr_en_buf_1_m2_ch16),
       .oe_0(oe_buf_0_m2_ch16),
       .oe_1(oe_buf_1_m2_ch16),
       .data_in_0(din_buf_0_m2_ch16),
       .data_in_1(din_buf_1_m2_ch16),
       .data_out_0(dout_buf_0_m2_ch16),
       .data_out_1(dout_buf_1_m2_ch16));
  
   always@(*)begin
       case(state)
           0:begin
               en_c1=0;en_m1=0;en_c2=0;en_m2=0;en_d1=0;en_sm=0;
               rst_c1=1;rst_m1=1;rst_c2=1;rst_m2=1;rst_d1=1;
           end
           1:begin
               en_c1=1;en_m1=0;en_c2=0;en_m2=0;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=1;rst_c2=1;rst_m2=1;rst_d1=1;
           end
           2:begin
               en_c1=0;en_m1=1;en_c2=0;en_m2=0;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=1;rst_m2=1;rst_d1=1;
           end
           3:begin
               en_c1=0;en_m1=0;en_c2=1;en_m2=0;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=0;rst_m2=1;rst_d1=1;
           end
           4:begin
               en_c1=0;en_m1=0;en_c2=0;en_m2=1;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=0;rst_m2=0;rst_d1=1;
           end
           5:begin
               en_c1=0;en_m1=0;en_c2=0;en_m2=0;en_d1=1;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=0;rst_m2=0;rst_d1=0;
           end
           6:begin
               en_c1=0;en_m1=0;en_c2=0;en_m2=0;en_d1=0;en_sm=1;
               rst_c1=0;rst_m1=0;rst_c2=0;rst_m2=0;rst_d1=0;
           end
           7:begin
               en_c1=1;en_m1=1;en_c2=0;en_m2=0;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=1;rst_m2=1;rst_d1=1;
           end
           8:begin
               en_c1=1;en_m1=0;en_c2=0;en_m2=0;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=0;rst_m2=1;rst_d1=1;
           end
           9:begin
               en_c1=0;en_m1=1;en_c2=1;en_m2=0;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=0;rst_m2=1;rst_d1=1;
           end
           10:begin
               en_c1=0;en_m1=1;en_c2=0;en_m2=0;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=0;rst_m2=1;rst_d1=1;
           end
           11:begin
               en_c1=1;en_m1=1;en_c2=1;en_m2=0;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=0;rst_m2=1;rst_d1=1;
           end
           12:begin
               en_c1=1;en_m1=0;en_c2=1;en_m2=0;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=0;rst_m2=1;rst_d1=1;
           end
           13:begin
               en_c1=0;en_m1=0;en_c2=1;en_m2=1;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=0;rst_m2=0;rst_d1=1;
           end
           14:begin
               en_c1=0;en_m1=0;en_c2=1;en_m2=0;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=0;rst_m2=0;rst_d1=1;
           end
           15:begin
               en_c1=1;en_m1=0;en_c2=1;en_m2=1;en_d1=0;en_sm=0;
               rst_c1=0;rst_m1=0;rst_c2=0;rst_m2=0;rst_d1=1;
           end
           default:begin
               en_c1=0;en_m1=0;en_c2=0;en_m2=0;en_d1=0;en_sm=0;
               rst_c1=1;rst_m1=1;rst_c2=1;rst_m2=1;rst_d1=0;
           end
      endcase
      if(done_soft==1)begin
          done=1'b1;
      end
      else begin
          prediction=4'dX;
          done=1'b0;
      end
  end
  
  
  integer j,k;
  always@(posedge clk)begin
          ///////////////////////////////////////// top /////////////////////////////////////////
      case(state)
          0:begin
              if(in_ready) state=1; else state=0;
          end
          1:begin
              if(done_c1) state=2; 
              else if(count_shift_c1>n*52-1) state=7;
              else state=1;
          end
          2:begin
              if(done_m1) state=3;
              else if(count_shift_m1>=(m+2)*13-1) state=9;
              else state=2;
          end
          3:begin
              if(done_c2) state=4;
              else if(count_shift_c2d>(a+a+1)*11-1) state=13;
              else state=3;
          end
          4:begin
              if(done_m2) state=5; else state=4;
          end
          5:begin
              if(done_den1) state=6; else state=5;
          end
          6:begin
              if(done_soft) state=0; else state=6;
          end
          7:begin
              if(done_mk_m1_row_ch1 && done_mk_m1_row_ch2 && done_mk_m1_row_ch3 && done_mk_m1_row_ch4 && done_mk_m1_row_ch5 && done_mk_m1_row_ch6) state=8;
              else state=7;
          end
          8:begin
              if(done_c1) state=2;
              else if(count_shift_c1>n*52-1) state=7;
              else if(count_shift_m1>=(m+2)*13-1) state=12;
              else state=8;
          end
          9:begin
              if(done_m1) state=3;
              else if(done_ck_c2d_row_ch1 && done_ck_c2d_row_ch2 && done_ck_c2d_row_ch3 && done_ck_c2d_row_ch4 && done_ck_c2d_row_ch5 && done_ck_c2d_row_ch6 && done_ck_c2d_row_ch7 && done_ck_c2d_row_ch8 && done_ck_c2d_row_ch9 && done_ck_c2d_row_ch10 && done_ck_c2d_row_ch11 && done_ck_c2d_row_ch12 && done_ck_c2d_row_ch13 && done_ck_c2d_row_ch14 && done_ck_c2d_row_ch15 && done_ck_c2d_row_ch16) state=10;
              else state=9;
          end
          10:begin
              if(done_m1) state=3;
              else if(count_shift_m1>=(m+2)*13-1) state=9;
              else state=10;
          end
          11:begin
              if(done_c1) state=2;
              else if(done_mk_m1_row_ch1 && done_mk_m1_row_ch2 && done_mk_m1_row_ch3 && done_mk_m1_row_ch4 && done_mk_m1_row_ch5 && done_mk_m1_row_ch6) state=8;
              else if(done_ck_c2d_row_ch1 && done_ck_c2d_row_ch2 && done_ck_c2d_row_ch3 && done_ck_c2d_row_ch4 && done_ck_c2d_row_ch5 && done_ck_c2d_row_ch6 && done_ck_c2d_row_ch7 && done_ck_c2d_row_ch8 && done_ck_c2d_row_ch9 && done_ck_c2d_row_ch10 && done_ck_c2d_row_ch11 && done_ck_c2d_row_ch12 && done_ck_c2d_row_ch13 && done_ck_c2d_row_ch14 && done_ck_c2d_row_ch15 && done_ck_c2d_row_ch16) state=7;
              else state=11;
          end
          12:begin
              if(done_c1) state=2;
              else if(done_ck_c2d_row_ch1 && done_ck_c2d_row_ch2 && done_ck_c2d_row_ch3 && done_ck_c2d_row_ch4 && done_ck_c2d_row_ch5 && done_ck_c2d_row_ch6 && done_ck_c2d_row_ch7 && done_ck_c2d_row_ch8 && done_ck_c2d_row_ch9 && done_ck_c2d_row_ch10 && done_ck_c2d_row_ch11 && done_ck_c2d_row_ch12 && done_ck_c2d_row_ch13 && done_ck_c2d_row_ch14 && done_ck_c2d_row_ch15 && done_ck_c2d_row_ch16) state=8;
              else if(count_shift_c2d>(a+a+1)*11-1) state=15;
              else state=12;
          end
          13:begin
              if(done_mk_m2_row_ch1 && done_mk_m2_row_ch2 && done_mk_m2_row_ch3 && done_mk_m2_row_ch4 && done_mk_m2_row_ch5 && done_mk_m2_row_ch6 && done_mk_m2_row_ch7 && done_mk_m2_row_ch8 && done_mk_m2_row_ch9 && done_mk_m2_row_ch10 && done_mk_m2_row_ch11 && done_mk_m2_row_ch12 && done_mk_m2_row_ch13 && done_mk_m2_row_ch14 && done_mk_m2_row_ch15 && done_mk_m2_row_ch16) state=14;
              else state=13;
          end
          14:begin
              if(done_c2) state=4;
              else if(count_shift_c2d>(a+a+1)*11-1) state=13;
              else state=14;
          end
          15:begin
              if(done_c1) state=2;
              else if(done_mk_m2_row_ch1 && done_mk_m2_row_ch2 && done_mk_m2_row_ch3 && done_mk_m2_row_ch4 && done_mk_m2_row_ch5 && done_mk_m2_row_ch6 && done_mk_m2_row_ch7 && done_mk_m2_row_ch8 && done_mk_m2_row_ch9 && done_mk_m2_row_ch10 && done_mk_m2_row_ch11 && done_mk_m2_row_ch12 && done_mk_m2_row_ch13 && done_mk_m2_row_ch14 && done_mk_m2_row_ch15 && done_mk_m2_row_ch16) state=12;
              else state=15;
          end
          default:begin
              state=0;
          end
      endcase
          //////////////////////////////////// c1 //////////////////////////////////
      if(rst_c1==0)begin
          case(state_c1)
              0:begin
                  if(en_c1) state_c1=2; else state_c1=0;
              end
              1:begin
                  if(done_shift_c1) state_c1=0; else state_c1=1;
              end
              2:begin
                  if(!done_shift_c1) state_c1=1; else state_c1=2;
              end
          endcase
      end
          //////////////////////////////shift_c1////////////////////////////
      if(rst_shift_c1)begin
          count_ld_shift_c1=0;
          done_load_shift_c1=0;
      end
      else if(en_shift_c1 && !done_load_shift_c1) begin
          wr_en=1'b0;
          oe=1'b1;
          if(count_ld_shift_c1<60)begin
              addr = OFFSET_C1_W+count_ld_shift_c1;
              count_ld_shift_c1 = count_ld_shift_c1+1;
              done_load_shift_c1=0;
          end
          else if(count_ld_shift_c1==60)begin
              done_load_shift_c1=0;
              count_ld_shift_c1 = count_ld_shift_c1+1;
          end
          else begin
              count_ld_shift_c1=11'hX;
              done_load_shift_c1=1'b1;
              wr_en=1'b0;
              oe=1'b0;
          end
      end
          /////////////////////////bram buffer c1////////////////////////
      if(rst_shift_c1==0 && done_load_shift_c1)begin
          if(count_shift_c1<676)begin
              addr_buf_0_c1_ch1=count_shift_c1;
              addr_buf_0_c1_ch2=count_shift_c1;
              addr_buf_0_c1_ch3=count_shift_c1;
              addr_buf_0_c1_ch4=count_shift_c1;
              addr_buf_0_c1_ch5=count_shift_c1;
              addr_buf_0_c1_ch6=count_shift_c1;
              wr_en_buf_0_c1_ch1=1'b1;
              wr_en_buf_0_c1_ch2=1'b1;
              wr_en_buf_0_c1_ch3=1'b1;
              wr_en_buf_0_c1_ch4=1'b1;
              wr_en_buf_0_c1_ch5=1'b1;
              wr_en_buf_0_c1_ch6=1'b1;
          end
          else begin
              addr_buf_0_c1_ch1=10'hX;
              addr_buf_0_c1_ch2=10'hX;
              addr_buf_0_c1_ch3=10'hX;
              addr_buf_0_c1_ch4=10'hX;
              addr_buf_0_c1_ch5=10'hX;
              addr_buf_0_c1_ch6=10'hX;
              wr_en_buf_0_c1_ch1=1'b0;
              wr_en_buf_0_c1_ch2=1'b0;
              wr_en_buf_0_c1_ch3=1'b0;
              wr_en_buf_0_c1_ch4=1'b0;
              wr_en_buf_0_c1_ch5=1'b0;
              wr_en_buf_0_c1_ch6=1'b0;
          end
      end
          ///////////////////////////// shift c1 //////////////////////////////
      if(rst_shift_c1==0 && done_load_shift_c1)begin
          case(state_shift_c1)
              0:begin
                  if(en_shift_c1 && done_load_shift_c1) state_shift_c1=2; else state_shift_c1=0;
              end
              1:begin
                  if(done_conv_c1) state_shift_c1=0; else state_shift_c1=1;
              end
              2:begin
                  if(!done_conv_c1) state_shift_c1=1; else state_shift_c1=2;
              end
          endcase
      end
          ////////////////////////conv_c1 channel 1 /////////////////////////////
      if(rst_conv_c1)begin
          count_ld_conv_c1=0;
          done_load_conv_c1=0;
      end
      else if(en_conv_c1 && !done_load_conv_c1) begin
          wr_en=1'b0;
          oe=1'b1;
          if(count_ld_conv_c1<3)begin
              addr = start_addr_conv_c1+count_ld_conv_c1;
              count_ld_conv_c1 = count_ld_conv_c1+1;
              done_load_conv_c1=0;
          end
          else if(count_ld_conv_c1<6)begin
              addr = start_addr_conv_c1+count_ld_conv_c1+25;
              count_ld_conv_c1 = count_ld_conv_c1+1;
              done_load_conv_c1=0;
          end
          else if(count_ld_conv_c1<9)begin
              addr = start_addr_conv_c1+count_ld_conv_c1+50;
              count_ld_conv_c1 = count_ld_conv_c1+1;
              done_load_conv_c1=0;
          end
          else if(count_ld_conv_c1==9)begin
              count_ld_conv_c1=count_ld_conv_c1+1;
              done_load_conv_c1=0;
          end
          else begin
             count_ld_conv_c1=11'hX;
             done_load_conv_c1=1'b1;
             wr_en=1'b0;
             oe=1'b0;
          end
      end
      if(rst_conv_c1==0 && done_load_conv_c1) begin
          case(state_conv_c1)
              0:begin
                  if(en_conv_c1 && done_load_conv_c1) state_conv_c1=2; else state_conv_c1=0;
              end
              1:begin
                  if(done_m_c1_ch1 && done_m_c1_ch2 && done_m_c1_ch3 && done_m_c1_ch4 && done_m_c1_ch5 && done_m_c1_ch6) state_conv_c1=0; else state_conv_c1=1;
              end
              2:begin
                  if(!done_m_c1_ch1 && !done_m_c1_ch2 && !done_m_c1_ch3 && !done_m_c1_ch4 && !done_m_c1_ch5 && !done_m_c1_ch6) state_conv_c1=1; else state_conv_c1=2;
              end
          endcase
      end
          ////////////////////////////maxpool m1/////////////////////////////
      if(rst_m1==0)begin
          case(state_m1)
              0:begin
                  if(en_m1) state_m1=2; else state_m1=0;
              end
              1:begin
                  if(done_shift_m1) state_m1=0;
                  else if(done_mk_m1_row_ch6) state_m1=3;
                  else state_m1=1;
              end
              2:begin
                  if(!done_mk_m1_row_ch1 && !done_mk_m1_row_ch2 && !done_mk_m1_row_ch3 && !done_mk_m1_row_ch4 && !done_mk_m1_row_ch5 && !done_mk_m1_row_ch6 && !done_shift_m1) state_m1=1; else state_m1=2;
              end
              3:begin
                  if(en_m1 && !done_mk_m1_row_ch1 && !done_mk_m1_row_ch2 && !done_mk_m1_row_ch3 && !done_mk_m1_row_ch4 && !done_mk_m1_row_ch5 && !done_mk_m1_row_ch6 && !done_shift_m1) state_m1=1;
                  else state_m1=3;
              end
          endcase
      end
          //////////////////////////// bram buffer m1 ///////////////////////////
      if(rst_shift_m1==0 )begin
          if(count_shift_m1<169)begin
              addr_buf_0_m1_ch1=count_shift_m1;
              addr_buf_0_m1_ch2=count_shift_m1;
              addr_buf_0_m1_ch3=count_shift_m1;
              addr_buf_0_m1_ch4=count_shift_m1;
              addr_buf_0_m1_ch5=count_shift_m1;
              addr_buf_0_m1_ch6=count_shift_m1;
              wr_en_buf_0_m1_ch1=1'b1;
              wr_en_buf_0_m1_ch2=1'b1;
              wr_en_buf_0_m1_ch3=1'b1;
              wr_en_buf_0_m1_ch4=1'b1;
              wr_en_buf_0_m1_ch5=1'b1;
              wr_en_buf_0_m1_ch6=1'b1;
          end
          else begin
              addr_buf_0_m1_ch1=8'hX;
              addr_buf_0_m1_ch2=8'hX;
              addr_buf_0_m1_ch3=8'hX;
              addr_buf_0_m1_ch4=8'hX;
              addr_buf_0_m1_ch5=8'hX;
              addr_buf_0_m1_ch6=8'hX;
              wr_en_buf_0_m1_ch1=1'b0;
              wr_en_buf_0_m1_ch2=1'b0;
              wr_en_buf_0_m1_ch3=1'b0;
              wr_en_buf_0_m1_ch4=1'b0;
              wr_en_buf_0_m1_ch5=1'b0;
              wr_en_buf_0_m1_ch6=1'b0;
          end
      end
          //////////////////////shift m1/////////////////////////
      if(rst_shift_m1==0)begin
          case(state_shift_m1)
              0:begin
                  if(en_shift_m1) state_shift_m1=2; else state_shift_m1=0;
              end
              1:begin
                  if(done_mk_m1_ch1 && done_mk_m1_ch2 && done_mk_m1_ch3 && done_mk_m1_ch4 && done_mk_m1_ch5 && done_mk_m1_ch6) state_shift_m1=0;
                  else if(done_mk_m1_row_ch1 && done_mk_m1_row_ch2 && done_mk_m1_row_ch3 && done_mk_m1_row_ch4 && done_mk_m1_row_ch5 && done_mk_m1_row_ch6) state_shift_m1=0;
                  else state_shift_m1=1;
              end
              2:begin
                  if(!done_mk_m1_ch1 && !done_mk_m1_row_ch1 && !done_mk_m1_ch2 && !done_mk_m1_row_ch2 && !done_mk_m1_ch3 && !done_mk_m1_row_ch3 && !done_mk_m1_ch4 && !done_mk_m1_row_ch4 && !done_mk_m1_ch5 && !done_mk_m1_row_ch5 && !done_mk_m1_ch6 && !done_mk_m1_row_ch6) state_shift_m1=1; else state_shift_m1=2;
              end
          endcase
      end
          //////////////////////////maxpool kernel m1////////////////////////////
      if(rst_mk_m1)begin
          count_ld_mk_m1=0;
          done_load_mk_m1=0;
      end
      else if(en_mk_m1 && !done_load_mk_m1) begin
          if(count_ld_mk_m1<2)begin
              wr_en_buf_1_c1_ch1=1'b0;
              wr_en_buf_1_c1_ch2=1'b0;
              wr_en_buf_1_c1_ch3=1'b0;
              wr_en_buf_1_c1_ch4=1'b0;
              wr_en_buf_1_c1_ch5=1'b0;
              wr_en_buf_1_c1_ch6=1'b0;
              oe_buf_1_c1_ch1=1'b1;
              oe_buf_1_c1_ch2=1'b1;
              oe_buf_1_c1_ch3=1'b1;
              oe_buf_1_c1_ch4=1'b1;
              oe_buf_1_c1_ch5=1'b1;
              oe_buf_1_c1_ch6=1'b1;
              addr_buf_1_c1_ch1 = start_addr_mk_m1+count_ld_mk_m1;
              addr_buf_1_c1_ch2 = start_addr_mk_m1+count_ld_mk_m1;
              addr_buf_1_c1_ch3 = start_addr_mk_m1+count_ld_mk_m1;
              addr_buf_1_c1_ch4 = start_addr_mk_m1+count_ld_mk_m1;
              addr_buf_1_c1_ch5 = start_addr_mk_m1+count_ld_mk_m1;
              addr_buf_1_c1_ch6 = start_addr_mk_m1+count_ld_mk_m1;
              count_ld_mk_m1 = count_ld_mk_m1+1;
              done_load_mk_m1=0;
          end
          else if(count_ld_mk_m1<4)begin
              wr_en_buf_1_c1_ch1=1'b0;
              wr_en_buf_1_c1_ch2=1'b0;
              wr_en_buf_1_c1_ch3=1'b0;
              wr_en_buf_1_c1_ch4=1'b0;
              wr_en_buf_1_c1_ch5=1'b0;
              wr_en_buf_1_c1_ch6=1'b0;
              oe_buf_1_c1_ch1=1'b1;
              oe_buf_1_c1_ch2=1'b1;
              oe_buf_1_c1_ch3=1'b1;
              oe_buf_1_c1_ch4=1'b1;
              oe_buf_1_c1_ch5=1'b1;
              oe_buf_1_c1_ch6=1'b1;
              addr_buf_1_c1_ch1 = start_addr_mk_m1+count_ld_mk_m1+24;
              addr_buf_1_c1_ch2 = start_addr_mk_m1+count_ld_mk_m1+24;
              addr_buf_1_c1_ch3 = start_addr_mk_m1+count_ld_mk_m1+24;
              addr_buf_1_c1_ch4 = start_addr_mk_m1+count_ld_mk_m1+24;
              addr_buf_1_c1_ch5 = start_addr_mk_m1+count_ld_mk_m1+24;
              addr_buf_1_c1_ch6 = start_addr_mk_m1+count_ld_mk_m1+24;
              count_ld_mk_m1 = count_ld_mk_m1+1;
              done_load_mk_m1=0;
          end
          else if(count_ld_mk_m1==4)begin
              count_ld_mk_m1=count_ld_mk_m1+1;
              done_load_mk_m1=0;
          end
          else begin
              count_ld_mk_m1=10'hX;
              done_load_mk_m1=1'b1;
          end
      end
          ///////////////////////////// maxpool m1 kernel calculation ///////////////////////////////
      if(rst_mk_m1)begin
          done_mk_m1_ch1=1'b0;
          done_mk_m1_ch2=1'b0;
          done_mk_m1_ch3=1'b0;
          done_mk_m1_ch4=1'b0;
          done_mk_m1_ch5=1'b0;
          done_mk_m1_ch6=1'b0;
          result_temp_mk_m1_ch1=0;
          result_temp_mk_m1_ch2=0;
          result_temp_mk_m1_ch3=0;
          result_temp_mk_m1_ch4=0;
          result_temp_mk_m1_ch5=0;
          result_temp_mk_m1_ch6=0;
          j=0;
          out_temp_mk_m1_ch1=0;
          out_temp_mk_m1_ch2=0;
          out_temp_mk_m1_ch3=0;
          out_temp_mk_m1_ch4=0;
          out_temp_mk_m1_ch5=0;
          out_temp_mk_m1_ch6=0;
      end
      else begin
          if(en_mk_m1==1 && done_load_mk_m1) begin
              for(j=0; j<4; j=j+1)begin
                  if(element_mk_m1_ch1[j]>out_temp_mk_m1_ch1)begin
                      out_temp_mk_m1_ch1=element_mk_m1_ch1[j];
                  end
                  if(element_mk_m1_ch2[j]>out_temp_mk_m1_ch2)begin
                      out_temp_mk_m1_ch2=element_mk_m1_ch2[j];
                  end
                  if(element_mk_m1_ch3[j]>out_temp_mk_m1_ch3)begin
                      out_temp_mk_m1_ch3=element_mk_m1_ch3[j];
                  end
                  if(element_mk_m1_ch4[j]>out_temp_mk_m1_ch4)begin
                      out_temp_mk_m1_ch4=element_mk_m1_ch4[j];
                  end
                  if(element_mk_m1_ch5[j]>out_temp_mk_m1_ch5)begin
                      out_temp_mk_m1_ch5=element_mk_m1_ch5[j];
                  end
                  if(element_mk_m1_ch6[j]>out_temp_mk_m1_ch6)begin
                      out_temp_mk_m1_ch6=element_mk_m1_ch6[j];
                  end
              end
              result_temp_mk_m1_ch1=out_temp_mk_m1_ch1;
              result_temp_mk_m1_ch2=out_temp_mk_m1_ch2;
              result_temp_mk_m1_ch3=out_temp_mk_m1_ch3;
              result_temp_mk_m1_ch4=out_temp_mk_m1_ch4;
              result_temp_mk_m1_ch5=out_temp_mk_m1_ch5;
              result_temp_mk_m1_ch6=out_temp_mk_m1_ch6;
              done_mk_m1_ch1=1'b1;
              done_mk_m1_ch2=1'b1;
              done_mk_m1_ch3=1'b1;
              done_mk_m1_ch4=1'b1;
              done_mk_m1_ch5=1'b1;
              done_mk_m1_ch6=1'b1;
          end
      end
          ///////////////////////////////// c2d /////////////////////////////////////
      if(rst_c2==0)begin
          bias_new_c2d_ch1=bias_c2d_ch1<<<3;
          bias_new_c2d_ch2=bias_c2d_ch2<<<3;
          bias_new_c2d_ch3=bias_c2d_ch3<<<3;
          bias_new_c2d_ch4=bias_c2d_ch4<<<3;
          bias_new_c2d_ch5=bias_c2d_ch5<<<3;
          bias_new_c2d_ch6=bias_c2d_ch6<<<3;
          bias_new_c2d_ch7=bias_c2d_ch7<<<3;
          bias_new_c2d_ch8=bias_c2d_ch8<<<3;
          bias_new_c2d_ch9=bias_c2d_ch9<<<3;
          bias_new_c2d_ch10=bias_c2d_ch10<<<3;
          bias_new_c2d_ch11=bias_c2d_ch11<<<3;
          bias_new_c2d_ch12=bias_c2d_ch12<<<3;
          bias_new_c2d_ch13=bias_c2d_ch13<<<3;
          bias_new_c2d_ch14=bias_c2d_ch14<<<3;
          bias_new_c2d_ch15=bias_c2d_ch15<<<3;
          bias_new_c2d_ch16=bias_c2d_ch16<<<3;
          case(state_c2d)
              0:begin
                  if(en_c2) state_c2d=2; else state_c2d=0;
              end
              1:begin
                  if(done_shift_c2d) state_c2d=0;
                  else if(done_ck_c2d_row_ch1 && done_ck_c2d_row_ch2 && done_ck_c2d_row_ch3 && done_ck_c2d_row_ch4 && done_ck_c2d_row_ch5 && done_ck_c2d_row_ch6 && done_ck_c2d_row_ch7 && done_ck_c2d_row_ch8 && done_ck_c2d_row_ch9 && done_ck_c2d_row_ch10 && done_ck_c2d_row_ch11 && done_ck_c2d_row_ch12 && done_ck_c2d_row_ch13 && done_ck_c2d_row_ch14 && done_ck_c2d_row_ch15 && done_ck_c2d_row_ch16) state_c2d=3;
                  else state_c2d=1;
              end
              2:begin
                  if(!done_shift_c2d && !done_ck_c2d_row_ch1 && !done_ck_c2d_row_ch2 && !done_ck_c2d_row_ch3 && !done_ck_c2d_row_ch4 && !done_ck_c2d_row_ch5 && !done_ck_c2d_row_ch6 && !done_ck_c2d_row_ch7 && !done_ck_c2d_row_ch8 && !done_ck_c2d_row_ch9 && !done_ck_c2d_row_ch10 && !done_ck_c2d_row_ch11 && !done_ck_c2d_row_ch12 && !done_ck_c2d_row_ch13 && !done_ck_c2d_row_ch14 && !done_ck_c2d_row_ch15 && !done_ck_c2d_row_ch16) state_c2d=1; else state_c2d=2;
              end
              3:begin
                  if(en_c2 && !done_shift_c2d && !done_ck_c2d_row_ch1 && !done_ck_c2d_row_ch2 && !done_ck_c2d_row_ch3 && !done_ck_c2d_row_ch4 && !done_ck_c2d_row_ch5 && !done_ck_c2d_row_ch6 && !done_ck_c2d_row_ch7 && !done_ck_c2d_row_ch8 && !done_ck_c2d_row_ch9 && !done_ck_c2d_row_ch10 && !done_ck_c2d_row_ch11 && !done_ck_c2d_row_ch12 && !done_ck_c2d_row_ch13 && !done_ck_c2d_row_ch14 && !done_ck_c2d_row_ch15 && !done_ck_c2d_row_ch16) state_c2d=1;
                  else state_c2d=3;
              end
          endcase
      end
         /////////////////////////////shift c2 ///////////////////////////////
      if(rst_shift_c2d)begin
          count_ld_shift_c2d=0;
          done_load_shift_c2d=0;
      end
      else if(en_shift_c2d && !done_load_shift_c2d) begin
          wr_en_c2=1'b0;
          oe_c2=1'b1;
          if(count_ld_shift_c2d<880)begin
              addr_c2 = OFFSET_C2_W+count_ld_shift_c2d;
              count_ld_shift_c2d = count_ld_shift_c2d+1;
              done_load_shift_c2d=0;
          end
          else if(count_ld_shift_c2d==880)
              count_ld_shift_c2d=count_ld_shift_c2d+1;
          else begin
              count_ld_shift_c2d=11'hX;
              done_load_shift_c2d=1'b1;
              wr_en_c2=1'b0;
              oe_c2=1'b0;
          end
      end
          ///////////////////////////// bram buffer c2 //////////////////////////////
      if(rst_shift_c2d==0 && done_load_shift_c2d)begin
          if(count_shift_c2d<121)begin
              addr_buf_0_c2_ch1=count_shift_c2d;
              addr_buf_0_c2_ch2=count_shift_c2d;
              addr_buf_0_c2_ch3=count_shift_c2d;
              addr_buf_0_c2_ch4=count_shift_c2d;
              addr_buf_0_c2_ch5=count_shift_c2d;
              addr_buf_0_c2_ch6=count_shift_c2d;
              addr_buf_0_c2_ch7=count_shift_c2d;
              addr_buf_0_c2_ch8=count_shift_c2d;
              addr_buf_0_c2_ch9=count_shift_c2d;
              addr_buf_0_c2_ch10=count_shift_c2d;
              addr_buf_0_c2_ch11=count_shift_c2d;
              addr_buf_0_c2_ch12=count_shift_c2d;
              addr_buf_0_c2_ch13=count_shift_c2d;
              addr_buf_0_c2_ch14=count_shift_c2d;
              addr_buf_0_c2_ch15=count_shift_c2d;
              addr_buf_0_c2_ch16=count_shift_c2d;
              wr_en_buf_0_c2_ch1=1'b1;
              wr_en_buf_0_c2_ch2=1'b1;
              wr_en_buf_0_c2_ch3=1'b1;
              wr_en_buf_0_c2_ch4=1'b1;
              wr_en_buf_0_c2_ch5=1'b1;
              wr_en_buf_0_c2_ch6=1'b1;
              wr_en_buf_0_c2_ch7=1'b1;
              wr_en_buf_0_c2_ch8=1'b1;
              wr_en_buf_0_c2_ch9=1'b1;
              wr_en_buf_0_c2_ch10=1'b1;
              wr_en_buf_0_c2_ch11=1'b1;
              wr_en_buf_0_c2_ch12=1'b1;
              wr_en_buf_0_c2_ch13=1'b1;
              wr_en_buf_0_c2_ch14=1'b1;
              wr_en_buf_0_c2_ch15=1'b1;
              wr_en_buf_0_c2_ch16=1'b1;
          end
          else begin
              addr_buf_0_c2_ch1=7'hX;
              addr_buf_0_c2_ch2=7'hX;
              addr_buf_0_c2_ch3=7'hX;
              addr_buf_0_c2_ch4=7'hX;
              addr_buf_0_c2_ch5=7'hX;
              addr_buf_0_c2_ch6=7'hX;
              addr_buf_0_c2_ch7=7'hX;
              addr_buf_0_c2_ch8=7'hX;
              addr_buf_0_c2_ch9=7'hX;
              addr_buf_0_c2_ch10=7'hX;
              addr_buf_0_c2_ch11=7'hX;
              addr_buf_0_c2_ch12=7'hX;
              addr_buf_0_c2_ch13=7'hX;
              addr_buf_0_c2_ch14=7'hX;
              addr_buf_0_c2_ch15=7'hX;
              addr_buf_0_c2_ch16=7'hX;
              wr_en_buf_0_c2_ch1=1'b0;
              wr_en_buf_0_c2_ch2=1'b0;
              wr_en_buf_0_c2_ch3=1'b0;
              wr_en_buf_0_c2_ch4=1'b0;
              wr_en_buf_0_c2_ch5=1'b0;
              wr_en_buf_0_c2_ch6=1'b0;
              wr_en_buf_0_c2_ch7=1'b0;
              wr_en_buf_0_c2_ch8=1'b0;
              wr_en_buf_0_c2_ch9=1'b0;
              wr_en_buf_0_c2_ch10=1'b0;
              wr_en_buf_0_c2_ch11=1'b0;
              wr_en_buf_0_c2_ch12=1'b0;
              wr_en_buf_0_c2_ch13=1'b0;
              wr_en_buf_0_c2_ch14=1'b0;
              wr_en_buf_0_c2_ch15=1'b0;
              wr_en_buf_0_c2_ch16=1'b0;
          end
      end
          ////////////////////////////// shift c2 ////////////////////////////////
      if(rst_shift_c2d==0 && done_load_shift_c2d)begin
          case(state_shift_c2d)
              0:begin
                  if(en_shift_c2d && done_load_shift_c2d) state_shift_c2d=2; else state_shift_c2d=0;
              end
              1:begin
                  if(done_ck_c2d) state_shift_c2d=0;
                  else if(done_ck_c2d_row_ch1 && done_ck_c2d_row_ch2 && done_ck_c2d_row_ch3 && done_ck_c2d_row_ch4 && done_ck_c2d_row_ch5 && done_ck_c2d_row_ch6 && done_ck_c2d_row_ch7 && done_ck_c2d_row_ch8 && done_ck_c2d_row_ch9 && done_ck_c2d_row_ch10 && done_ck_c2d_row_ch11 && done_ck_c2d_row_ch12 && done_ck_c2d_row_ch13 && done_ck_c2d_row_ch14 && done_ck_c2d_row_ch15 && done_ck_c2d_row_ch16) state_shift_c2d=0;
                  else state_shift_c2d=1;
              end
              2:begin
                  if(!done_ck_c2d && !done_ck_c2d_row_ch1 && !done_ck_c2d_row_ch2 && !done_ck_c2d_row_ch3 && !done_ck_c2d_row_ch4 && !done_ck_c2d_row_ch5 && !done_ck_c2d_row_ch6 && !done_ck_c2d_row_ch7 && !done_ck_c2d_row_ch8 && !done_ck_c2d_row_ch9 && !done_ck_c2d_row_ch10 && !done_ck_c2d_row_ch11 && !done_ck_c2d_row_ch12 && !done_ck_c2d_row_ch13 && !done_ck_c2d_row_ch14 && !done_ck_c2d_row_ch15 && !done_ck_c2d_row_ch16) state_shift_c2d=1; else state_shift_c2d=2;
              end
          endcase
      end
          ////////////////////////////////conv kernel c2 /////////////////////////////////////
      if(rst_ck_c2d)begin
          count_ld_conv_c2=0;
          done_load_conv_c2=0;
      end
      else if(en_ck_c2d && !done_load_conv_c2) begin
          if(count_ld_conv_c2<3)begin
              wr_en_buf_1_m1_ch1=1'b0;
              wr_en_buf_1_m1_ch2=1'b0;
              wr_en_buf_1_m1_ch3=1'b0;
              wr_en_buf_1_m1_ch4=1'b0;
              wr_en_buf_1_m1_ch5=1'b0;
              wr_en_buf_1_m1_ch6=1'b0;
              oe_buf_1_m1_ch1=1'b1;
              oe_buf_1_m1_ch2=1'b1;
              oe_buf_1_m1_ch3=1'b1;
              oe_buf_1_m1_ch4=1'b1;
              oe_buf_1_m1_ch5=1'b1;
              oe_buf_1_m1_ch6=1'b1;
              addr_buf_1_m1_ch1 = start_addr_conv_c2+count_ld_conv_c2;
              addr_buf_1_m1_ch2 = start_addr_conv_c2+count_ld_conv_c2;
              addr_buf_1_m1_ch3 = start_addr_conv_c2+count_ld_conv_c2;
              addr_buf_1_m1_ch4 = start_addr_conv_c2+count_ld_conv_c2;
              addr_buf_1_m1_ch5 = start_addr_conv_c2+count_ld_conv_c2;
              addr_buf_1_m1_ch6 = start_addr_conv_c2+count_ld_conv_c2;
              count_ld_conv_c2 = count_ld_conv_c2+1;
              done_load_conv_c2=0;
          end
          else if(count_ld_conv_c2<6)begin
              wr_en_buf_1_m1_ch1=1'b0;
              wr_en_buf_1_m1_ch2=1'b0;
              wr_en_buf_1_m1_ch3=1'b0;
              wr_en_buf_1_m1_ch4=1'b0;
              wr_en_buf_1_m1_ch5=1'b0;
              wr_en_buf_1_m1_ch6=1'b0;
              oe_buf_1_m1_ch1=1'b1;
              oe_buf_1_m1_ch2=1'b1;
              oe_buf_1_m1_ch3=1'b1;
              oe_buf_1_m1_ch4=1'b1;
              oe_buf_1_m1_ch5=1'b1;
              oe_buf_1_m1_ch6=1'b1;
              addr_buf_1_m1_ch1 = start_addr_conv_c2+count_ld_conv_c2+10;
              addr_buf_1_m1_ch2 = start_addr_conv_c2+count_ld_conv_c2+10;
              addr_buf_1_m1_ch3 = start_addr_conv_c2+count_ld_conv_c2+10;
              addr_buf_1_m1_ch4 = start_addr_conv_c2+count_ld_conv_c2+10;
              addr_buf_1_m1_ch5 = start_addr_conv_c2+count_ld_conv_c2+10;
              addr_buf_1_m1_ch6 = start_addr_conv_c2+count_ld_conv_c2+10;
              count_ld_conv_c2 = count_ld_conv_c2+1;
              done_load_conv_c2=0;
          end
          else if(count_ld_conv_c2<9)begin
              wr_en_buf_1_m1_ch1=1'b0;
              wr_en_buf_1_m1_ch2=1'b0;
              wr_en_buf_1_m1_ch3=1'b0;
              wr_en_buf_1_m1_ch4=1'b0;
              wr_en_buf_1_m1_ch5=1'b0;
              wr_en_buf_1_m1_ch6=1'b0;
              oe_buf_1_m1_ch1=1'b1;
              oe_buf_1_m1_ch2=1'b1;
              oe_buf_1_m1_ch3=1'b1;
              oe_buf_1_m1_ch4=1'b1;
              oe_buf_1_m1_ch5=1'b1;
              oe_buf_1_m1_ch6=1'b1;
              addr_buf_1_m1_ch1 = start_addr_conv_c2+count_ld_conv_c2+20;
              addr_buf_1_m1_ch2 = start_addr_conv_c2+count_ld_conv_c2+20;
              addr_buf_1_m1_ch3 = start_addr_conv_c2+count_ld_conv_c2+20;
              addr_buf_1_m1_ch4 = start_addr_conv_c2+count_ld_conv_c2+20;
              addr_buf_1_m1_ch5 = start_addr_conv_c2+count_ld_conv_c2+20;
              addr_buf_1_m1_ch6 = start_addr_conv_c2+count_ld_conv_c2+20;
              count_ld_conv_c2 = count_ld_conv_c2+1;
              done_load_conv_c2=0;
          end
          else if(count_ld_conv_c2==9)begin
              count_ld_conv_c2=count_ld_conv_c2+1;
              done_load_conv_c2=0;
          end
          else begin
              count_ld_conv_c2=11'hX;
              done_load_conv_c2=1'b1;
          end
      end
      if(rst_ck_c2d==0 && done_load_conv_c2) begin
          case(state_ck_c2d)
              0:begin
                  if(en_ck_c2d && done_load_conv_c2) state_ck_c2d=2; else state_ck_c2d=0;
              end
              1:begin
                  if(done_m_c2d_in1_w11 && done_m_c2d_in2_w12 && done_m_c2d_in3_w13 && done_m_c2d_in4_w14 && done_m_c2d_in5_w15 && done_m_c2d_in6_w16 && done_m_c2d_in1_w21 && done_m_c2d_in2_w22 && done_m_c2d_in3_w23 && done_m_c2d_in4_w24 && done_m_c2d_in5_w25 && done_m_c2d_in6_w26 && done_m_c2d_in1_w31 && done_m_c2d_in2_w32 && done_m_c2d_in3_w33 && done_m_c2d_in4_w34 && done_m_c2d_in5_w35 && done_m_c2d_in6_w36 && done_m_c2d_in1_w41 && done_m_c2d_in2_w42 && done_m_c2d_in3_w43 && done_m_c2d_in4_w44 && done_m_c2d_in5_w45 && done_m_c2d_in6_w46 && done_m_c2d_in1_w51 && done_m_c2d_in2_w52 && done_m_c2d_in3_w53 && done_m_c2d_in4_w54 && done_m_c2d_in5_w55 && done_m_c2d_in6_w56 && done_m_c2d_in1_w61 && done_m_c2d_in2_w62 && done_m_c2d_in3_w63 && done_m_c2d_in4_w64 && done_m_c2d_in5_w65 && done_m_c2d_in6_w66 && done_m_c2d_in1_w71 && done_m_c2d_in2_w72 && done_m_c2d_in3_w73 && done_m_c2d_in4_w74 && done_m_c2d_in5_w75 && done_m_c2d_in6_w76 && done_m_c2d_in1_w81 && done_m_c2d_in2_w82 && done_m_c2d_in3_w83 && done_m_c2d_in4_w84 && done_m_c2d_in5_w85 && done_m_c2d_in6_w86 && done_m_c2d_in1_w91 && done_m_c2d_in2_w92 && done_m_c2d_in3_w93 && done_m_c2d_in4_w94 && done_m_c2d_in5_w95 && done_m_c2d_in6_w96 && done_m_c2d_in1_w101 && done_m_c2d_in2_w102 && done_m_c2d_in3_w103 && done_m_c2d_in4_w104 && done_m_c2d_in5_w105 && done_m_c2d_in6_w106 && done_m_c2d_in1_w111 && done_m_c2d_in2_w112 && done_m_c2d_in3_w113 && done_m_c2d_in4_w114 && done_m_c2d_in5_w115 && done_m_c2d_in6_w116 && done_m_c2d_in1_w121 && done_m_c2d_in2_w122 && done_m_c2d_in3_w123 && done_m_c2d_in4_w124 && done_m_c2d_in5_w125 && done_m_c2d_in6_w126 && done_m_c2d_in1_w131 && done_m_c2d_in2_w132 && done_m_c2d_in3_w133 && done_m_c2d_in4_w134 && done_m_c2d_in5_w135 && done_m_c2d_in6_w136 && done_m_c2d_in1_w141 && done_m_c2d_in2_w142 && done_m_c2d_in3_w143 && done_m_c2d_in4_w144 && done_m_c2d_in5_w145 && done_m_c2d_in6_w146 && done_m_c2d_in1_w151 && done_m_c2d_in2_w152 && done_m_c2d_in3_w153 && done_m_c2d_in4_w154 && done_m_c2d_in5_w155 && done_m_c2d_in6_w156 && done_m_c2d_in1_w161 && done_m_c2d_in2_w162 && done_m_c2d_in3_w163 && done_m_c2d_in4_w164 && done_m_c2d_in5_w165 && done_m_c2d_in6_w166) state_ck_c2d=0; else state_ck_c2d=1;
              end
              2:begin
                  if(!(done_m_c2d_in1_w11 && done_m_c2d_in2_w12 && done_m_c2d_in3_w13 && done_m_c2d_in4_w14 && done_m_c2d_in5_w15 && done_m_c2d_in6_w16 && done_m_c2d_in1_w21 && done_m_c2d_in2_w22 && done_m_c2d_in3_w23 && done_m_c2d_in4_w24 && done_m_c2d_in5_w25 && done_m_c2d_in6_w26 && done_m_c2d_in1_w31 && done_m_c2d_in2_w32 && done_m_c2d_in3_w33 && done_m_c2d_in4_w34 && done_m_c2d_in5_w35 && done_m_c2d_in6_w36 && done_m_c2d_in1_w41 && done_m_c2d_in2_w42 && done_m_c2d_in3_w43 && done_m_c2d_in4_w44 && done_m_c2d_in5_w45 && done_m_c2d_in6_w46 && done_m_c2d_in1_w51 && done_m_c2d_in2_w52 && done_m_c2d_in3_w53 && done_m_c2d_in4_w54 && done_m_c2d_in5_w55 && done_m_c2d_in6_w56 && done_m_c2d_in1_w61 && done_m_c2d_in2_w62 && done_m_c2d_in3_w63 && done_m_c2d_in4_w64 && done_m_c2d_in5_w65 && done_m_c2d_in6_w66 && done_m_c2d_in1_w71 && done_m_c2d_in2_w72 && done_m_c2d_in3_w73 && done_m_c2d_in4_w74 && done_m_c2d_in5_w75 && done_m_c2d_in6_w76 && done_m_c2d_in1_w81 && done_m_c2d_in2_w82 && done_m_c2d_in3_w83 && done_m_c2d_in4_w84 && done_m_c2d_in5_w85 && done_m_c2d_in6_w86 && done_m_c2d_in1_w91 && done_m_c2d_in2_w92 && done_m_c2d_in3_w93 && done_m_c2d_in4_w94 && done_m_c2d_in5_w95 && done_m_c2d_in6_w96 && done_m_c2d_in1_w101 && done_m_c2d_in2_w102 && done_m_c2d_in3_w103 && done_m_c2d_in4_w104 && done_m_c2d_in5_w105 && done_m_c2d_in6_w106 && done_m_c2d_in1_w111 && done_m_c2d_in2_w112 && done_m_c2d_in3_w113 && done_m_c2d_in4_w114 && done_m_c2d_in5_w115 && done_m_c2d_in6_w116 && done_m_c2d_in1_w121 && done_m_c2d_in2_w122 && done_m_c2d_in3_w123 && done_m_c2d_in4_w124 && done_m_c2d_in5_w125 && done_m_c2d_in6_w126 && done_m_c2d_in1_w131 && done_m_c2d_in2_w132 && done_m_c2d_in3_w133 && done_m_c2d_in4_w134 && done_m_c2d_in5_w135 && done_m_c2d_in6_w136 && done_m_c2d_in1_w141 && done_m_c2d_in2_w142 && done_m_c2d_in3_w143 && done_m_c2d_in4_w144 && done_m_c2d_in5_w145 && done_m_c2d_in6_w146 && done_m_c2d_in1_w151 && done_m_c2d_in2_w152 && done_m_c2d_in3_w153 && done_m_c2d_in4_w154 && done_m_c2d_in5_w155 && done_m_c2d_in6_w156 && done_m_c2d_in1_w161 && done_m_c2d_in2_w162 && done_m_c2d_in3_w163 && done_m_c2d_in4_w164 && done_m_c2d_in5_w165 && done_m_c2d_in6_w166)) state_ck_c2d=1; else state_ck_c2d=2;
              end
              default:begin
                  state_ck_c2d=0;
              end
          endcase
      end
          ///////////////////////////// mapool m2 /////////////////////////////
      if(rst_m2==0)begin
          case(state_m2)
              0:begin
                  if(en_m2) state_m2=2; else state_m2=0;
              end
              1:begin
                  if(done_shift_m2) state_m2=0;
                  else if(done_mk_m2_row_ch1 && done_mk_m2_row_ch2 && done_mk_m2_row_ch3 && done_mk_m2_row_ch4 && done_mk_m2_row_ch5 && done_mk_m2_row_ch6 && done_mk_m2_row_ch7 && done_mk_m2_row_ch8 && done_mk_m2_row_ch9 && done_mk_m2_row_ch10 && done_mk_m2_row_ch11 && done_mk_m2_row_ch12 && done_mk_m2_row_ch13 && done_mk_m2_row_ch14 && done_mk_m2_row_ch15 && done_mk_m2_row_ch16) state_m2=3;
                  else state_m2=1;
              end
              2:begin
                  if(!done_shift_m2 && !done_mk_m2_row_ch1 && !done_mk_m2_row_ch2 && !done_mk_m2_row_ch3 && !done_mk_m2_row_ch4 && !done_mk_m2_row_ch5 && !done_mk_m2_row_ch6 && !done_mk_m2_row_ch7 && !done_mk_m2_row_ch8 && !done_mk_m2_row_ch9 && !done_mk_m2_row_ch10 && !done_mk_m2_row_ch11 && !done_mk_m2_row_ch12 && !done_mk_m2_row_ch13 && !done_mk_m2_row_ch14 && !done_mk_m2_row_ch15 && !done_mk_m2_row_ch16) state_m2=1; else state_m2=2;
              end
              3:begin
                  if(en_m2 && !done_mk_m2_row_ch1 && !done_mk_m2_row_ch2 && !done_mk_m2_row_ch3 && !done_mk_m2_row_ch4 && !done_mk_m2_row_ch5 && !done_mk_m2_row_ch6 && !done_mk_m2_row_ch7 && !done_mk_m2_row_ch8 && !done_mk_m2_row_ch9 && !done_mk_m2_row_ch10 && !done_mk_m2_row_ch11 && !done_mk_m2_row_ch12 && !done_mk_m2_row_ch13 && !done_mk_m2_row_ch14 && !done_mk_m2_row_ch15 && !done_mk_m2_row_ch16 && !done_shift_m2) state_m2=1;
                  else state_m2=3;
              end
          endcase
      end
          ////////////////////////////// bram buffer m2 //////////////////////////////
      if(rst_shift_m2==0 )begin
          if(count_shift_m2<25)begin
              addr_buf_0_m2_ch1=count_shift_m2;
              addr_buf_0_m2_ch2=count_shift_m2;
              addr_buf_0_m2_ch3=count_shift_m2;
              addr_buf_0_m2_ch4=count_shift_m2;
              addr_buf_0_m2_ch5=count_shift_m2;
              addr_buf_0_m2_ch6=count_shift_m2;
              addr_buf_0_m2_ch7=count_shift_m2;
              addr_buf_0_m2_ch8=count_shift_m2;
              addr_buf_0_m2_ch9=count_shift_m2;
              addr_buf_0_m2_ch10=count_shift_m2;
              addr_buf_0_m2_ch11=count_shift_m2;
              addr_buf_0_m2_ch12=count_shift_m2;
              addr_buf_0_m2_ch13=count_shift_m2;
              addr_buf_0_m2_ch14=count_shift_m2;
              addr_buf_0_m2_ch15=count_shift_m2;
              addr_buf_0_m2_ch16=count_shift_m2;
              wr_en_buf_0_m2_ch1=1'b1;
              wr_en_buf_0_m2_ch2=1'b1;
              wr_en_buf_0_m2_ch3=1'b1;
              wr_en_buf_0_m2_ch4=1'b1;
              wr_en_buf_0_m2_ch5=1'b1;
              wr_en_buf_0_m2_ch6=1'b1;
              wr_en_buf_0_m2_ch7=1'b1;
              wr_en_buf_0_m2_ch8=1'b1;
              wr_en_buf_0_m2_ch9=1'b1;
              wr_en_buf_0_m2_ch10=1'b1;
              wr_en_buf_0_m2_ch11=1'b1;
              wr_en_buf_0_m2_ch12=1'b1;
              wr_en_buf_0_m2_ch13=1'b1;
              wr_en_buf_0_m2_ch14=1'b1;
              wr_en_buf_0_m2_ch15=1'b1;
              wr_en_buf_0_m2_ch16=1'b1;
          end
          else begin
              addr_buf_0_m2_ch1=5'hX;
              addr_buf_0_m2_ch2=5'hX;
              addr_buf_0_m2_ch3=5'hX;
              addr_buf_0_m2_ch4=5'hX;
              addr_buf_0_m2_ch5=5'hX;
              addr_buf_0_m2_ch6=5'hX;
              addr_buf_0_m2_ch7=5'hX;
              addr_buf_0_m2_ch8=5'hX;
              addr_buf_0_m2_ch9=5'hX;
              addr_buf_0_m2_ch10=5'hX;
              addr_buf_0_m2_ch11=5'hX;
              addr_buf_0_m2_ch12=5'hX;
              addr_buf_0_m2_ch13=5'hX;
              addr_buf_0_m2_ch14=5'hX;
              addr_buf_0_m2_ch15=5'hX;
              addr_buf_0_m2_ch16=5'hX;
              wr_en_buf_0_m2_ch1=1'b0;
              wr_en_buf_0_m2_ch2=1'b0;
              wr_en_buf_0_m2_ch3=1'b0;
              wr_en_buf_0_m2_ch4=1'b0;
              wr_en_buf_0_m2_ch5=1'b0;
              wr_en_buf_0_m2_ch6=1'b0;
              wr_en_buf_0_m2_ch7=1'b0;
              wr_en_buf_0_m2_ch8=1'b0;
              wr_en_buf_0_m2_ch9=1'b0;
              wr_en_buf_0_m2_ch10=1'b0;
              wr_en_buf_0_m2_ch11=1'b0;
              wr_en_buf_0_m2_ch12=1'b0;
              wr_en_buf_0_m2_ch13=1'b0;
              wr_en_buf_0_m2_ch14=1'b0;
              wr_en_buf_0_m2_ch15=1'b0;
              wr_en_buf_0_m2_ch16=1'b0;
          end
      end
          //////////////////////////////// shift window m2 /////////////////////////////////
      if(rst_shift_m2==0)begin
          case(state_shift_m2)
              0:begin
                  if(en_shift_m2) state_shift_m2=2; else state_shift_m2=0;
              end
              1:begin
                  if(done_mk_m2) state_shift_m2=0;
                  else if(done_mk_m2_row_ch1 && done_mk_m2_row_ch2 && done_mk_m2_row_ch3 && done_mk_m2_row_ch4 && done_mk_m2_row_ch5 && done_mk_m2_row_ch6 && done_mk_m2_row_ch7 && done_mk_m2_row_ch8 && done_mk_m2_row_ch9 && done_mk_m2_row_ch10 && done_mk_m2_row_ch11 && done_mk_m2_row_ch12 && done_mk_m2_row_ch13 && done_mk_m2_row_ch14 && done_mk_m2_row_ch15 && done_mk_m2_row_ch16) state_shift_m2=0;
                  else state_shift_m2=1;
              end
              2:begin
                  if(!done_mk_m2 && !done_mk_m2_row_ch1 && !done_mk_m2_row_ch2 && !done_mk_m2_row_ch3 && !done_mk_m2_row_ch4 && !done_mk_m2_row_ch5 && !done_mk_m2_row_ch6 && !done_mk_m2_row_ch7 && !done_mk_m2_row_ch8 && !done_mk_m2_row_ch9 && !done_mk_m2_row_ch10 && !done_mk_m2_row_ch11 && !done_mk_m2_row_ch12 && !done_mk_m2_row_ch13 && !done_mk_m2_row_ch14 && !done_mk_m2_row_ch15 && !done_mk_m2_row_ch16) state_shift_m2=1; else state_shift_m2=2;
              end
          endcase
      end
          /////////////////////////////// maxpool kernel m2 ///////////////////////////////
      if(rst_mk_m2)begin
          count_ld_mk_m2=0;
          done_load_mk_m2=0;
      end
      else if(en_mk_m2 && !done_load_mk_m2) begin
          if(count_ld_mk_m2<3)begin
              wr_en_buf_1_c2_ch1=1'b0;
              wr_en_buf_1_c2_ch2=1'b0;
              wr_en_buf_1_c2_ch3=1'b0;
              wr_en_buf_1_c2_ch4=1'b0;
              wr_en_buf_1_c2_ch5=1'b0;
              wr_en_buf_1_c2_ch6=1'b0;
              wr_en_buf_1_c2_ch7=1'b0;
              wr_en_buf_1_c2_ch8=1'b0;
              wr_en_buf_1_c2_ch9=1'b0;
              wr_en_buf_1_c2_ch10=1'b0;
              wr_en_buf_1_c2_ch11=1'b0;
              wr_en_buf_1_c2_ch12=1'b0;
              wr_en_buf_1_c2_ch13=1'b0;
              wr_en_buf_1_c2_ch14=1'b0;
              wr_en_buf_1_c2_ch15=1'b0;
              wr_en_buf_1_c2_ch16=1'b0;
              oe_buf_1_c2_ch1=1'b1;
              oe_buf_1_c2_ch2=1'b1;
              oe_buf_1_c2_ch3=1'b1;
              oe_buf_1_c2_ch4=1'b1;
              oe_buf_1_c2_ch5=1'b1;
              oe_buf_1_c2_ch6=1'b1;
              oe_buf_1_c2_ch7=1'b1;
              oe_buf_1_c2_ch8=1'b1;
              oe_buf_1_c2_ch9=1'b1;
              oe_buf_1_c2_ch10=1'b1;
              oe_buf_1_c2_ch11=1'b1;
              oe_buf_1_c2_ch12=1'b1;
              oe_buf_1_c2_ch13=1'b1;
              oe_buf_1_c2_ch14=1'b1;
              oe_buf_1_c2_ch15=1'b1;
              oe_buf_1_c2_ch16=1'b1;
              addr_buf_1_c2_ch1 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch2 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch3 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch4 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch5 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch6 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch7 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch8 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch9 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch10 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch11 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch12 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch13 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch14 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch15 = start_addr_mk_m2+count_ld_mk_m2;
              addr_buf_1_c2_ch16 = start_addr_mk_m2+count_ld_mk_m2;
              count_ld_mk_m2 = count_ld_mk_m2+1;
              done_load_mk_m2=0;
          end
          else if(count_ld_mk_m2<6)begin
              wr_en_buf_1_c2_ch1=1'b0;
              wr_en_buf_1_c2_ch2=1'b0;
              wr_en_buf_1_c2_ch3=1'b0;
              wr_en_buf_1_c2_ch4=1'b0;
              wr_en_buf_1_c2_ch5=1'b0;
              wr_en_buf_1_c2_ch6=1'b0;
              wr_en_buf_1_c2_ch7=1'b0;
              wr_en_buf_1_c2_ch8=1'b0;
              wr_en_buf_1_c2_ch9=1'b0;
              wr_en_buf_1_c2_ch10=1'b0;
              wr_en_buf_1_c2_ch11=1'b0;
              wr_en_buf_1_c2_ch12=1'b0;
              wr_en_buf_1_c2_ch13=1'b0;
              wr_en_buf_1_c2_ch14=1'b0;
              wr_en_buf_1_c2_ch15=1'b0;
              wr_en_buf_1_c2_ch16=1'b0;
              oe_buf_1_c2_ch1=1'b1;
              oe_buf_1_c2_ch2=1'b1;
              oe_buf_1_c2_ch3=1'b1;
              oe_buf_1_c2_ch4=1'b1;
              oe_buf_1_c2_ch5=1'b1;
              oe_buf_1_c2_ch6=1'b1;
              oe_buf_1_c2_ch7=1'b1;
              oe_buf_1_c2_ch8=1'b1;
              oe_buf_1_c2_ch9=1'b1;
              oe_buf_1_c2_ch10=1'b1;
              oe_buf_1_c2_ch11=1'b1;
              oe_buf_1_c2_ch12=1'b1;
              oe_buf_1_c2_ch13=1'b1;
              oe_buf_1_c2_ch14=1'b1;
              oe_buf_1_c2_ch15=1'b1;
              oe_buf_1_c2_ch16=1'b1;
              addr_buf_1_c2_ch1 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch2 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch3 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch4 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch5 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch6 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch7 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch8 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch9 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch10 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch11 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch12 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch13 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch14 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch15 = start_addr_mk_m2+count_ld_mk_m2+8;
              addr_buf_1_c2_ch16 = start_addr_mk_m2+count_ld_mk_m2+8;
              count_ld_mk_m2 = count_ld_mk_m2+1;
              done_load_mk_m1=0;
          end
          else if(count_ld_mk_m2<9)begin
              wr_en_buf_1_c2_ch1=1'b0;
              wr_en_buf_1_c2_ch2=1'b0;
              wr_en_buf_1_c2_ch3=1'b0;
              wr_en_buf_1_c2_ch4=1'b0;
              wr_en_buf_1_c2_ch5=1'b0;
              wr_en_buf_1_c2_ch6=1'b0;
              wr_en_buf_1_c2_ch7=1'b0;
              wr_en_buf_1_c2_ch8=1'b0;
              wr_en_buf_1_c2_ch9=1'b0;
              wr_en_buf_1_c2_ch10=1'b0;
              wr_en_buf_1_c2_ch11=1'b0;
              wr_en_buf_1_c2_ch12=1'b0;
              wr_en_buf_1_c2_ch13=1'b0;
              wr_en_buf_1_c2_ch14=1'b0;
              wr_en_buf_1_c2_ch15=1'b0;
              wr_en_buf_1_c2_ch16=1'b0;
              oe_buf_1_c2_ch1=1'b1;
              oe_buf_1_c2_ch2=1'b1;
              oe_buf_1_c2_ch3=1'b1;
              oe_buf_1_c2_ch4=1'b1;
              oe_buf_1_c2_ch5=1'b1;
              oe_buf_1_c2_ch6=1'b1;
              oe_buf_1_c2_ch7=1'b1;
              oe_buf_1_c2_ch8=1'b1;
              oe_buf_1_c2_ch9=1'b1;
              oe_buf_1_c2_ch10=1'b1;
              oe_buf_1_c2_ch11=1'b1;
              oe_buf_1_c2_ch12=1'b1;
              oe_buf_1_c2_ch13=1'b1;
              oe_buf_1_c2_ch14=1'b1;
              oe_buf_1_c2_ch15=1'b1;
              oe_buf_1_c2_ch16=1'b1;
              addr_buf_1_c2_ch1 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch2 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch3 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch4 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch5 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch6 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch7 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch8 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch9 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch10 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch11 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch12 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch13 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch14 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch15 = start_addr_mk_m2+count_ld_mk_m2+16;
              addr_buf_1_c2_ch16 = start_addr_mk_m2+count_ld_mk_m2+16;
              count_ld_mk_m2 = count_ld_mk_m2+1;
              done_load_mk_m1=0;
          end
          else if(count_ld_mk_m2==9)begin
              count_ld_mk_m2=count_ld_mk_m2+1;
              done_load_mk_m2=0;
          end
          else begin
              count_ld_mk_m2=10'hX;
              done_load_mk_m2=1'b1;
          end
      end
      if(rst_mk_m2)begin
          done_mk_m2=1'b0;
          result_temp_mk_m2_ch1=0;
          result_temp_mk_m2_ch2=0;
          result_temp_mk_m2_ch3=0;
          result_temp_mk_m2_ch4=0;
          result_temp_mk_m2_ch5=0;
          result_temp_mk_m2_ch6=0;
          result_temp_mk_m2_ch7=0;
          result_temp_mk_m2_ch8=0;
          result_temp_mk_m2_ch9=0;
          result_temp_mk_m2_ch10=0;
          result_temp_mk_m2_ch11=0;
          result_temp_mk_m2_ch12=0;
          result_temp_mk_m2_ch13=0;
          result_temp_mk_m2_ch14=0;
          result_temp_mk_m2_ch15=0;
          result_temp_mk_m2_ch16=0;
          k=0;
          out_temp_mk_m2_ch1=0;
          out_temp_mk_m2_ch2=0;
          out_temp_mk_m2_ch3=0;
          out_temp_mk_m2_ch4=0;
          out_temp_mk_m2_ch5=0;
          out_temp_mk_m2_ch6=0;
          out_temp_mk_m2_ch7=0;
          out_temp_mk_m2_ch8=0;
          out_temp_mk_m2_ch9=0;
          out_temp_mk_m2_ch10=0;
          out_temp_mk_m2_ch11=0;
          out_temp_mk_m2_ch12=0;
          out_temp_mk_m2_ch13=0;
          out_temp_mk_m2_ch14=0;
          out_temp_mk_m2_ch15=0;
          out_temp_mk_m2_ch16=0;
      end
      else begin
          if(en_mk_m2==1 && done_load_mk_m2) begin
              for(k=0; k<9; k=k+1)begin
                  if(element_mk_m2_ch1[k]>out_temp_mk_m2_ch1)begin
                      out_temp_mk_m2_ch1=element_mk_m2_ch1[k];
                  end
                  if(element_mk_m2_ch2[k]>out_temp_mk_m2_ch2)begin
                      out_temp_mk_m2_ch2=element_mk_m2_ch2[k];
                  end
                  if(element_mk_m2_ch3[k]>out_temp_mk_m2_ch3)begin
                      out_temp_mk_m2_ch3=element_mk_m2_ch3[k];
                  end
                  if(element_mk_m2_ch4[k]>out_temp_mk_m2_ch4)begin
                      out_temp_mk_m2_ch4=element_mk_m2_ch4[k];
                  end
                  if(element_mk_m2_ch5[k]>out_temp_mk_m2_ch5)begin
                      out_temp_mk_m2_ch5=element_mk_m2_ch5[k];
                  end
                  if(element_mk_m2_ch6[k]>out_temp_mk_m2_ch6)begin
                      out_temp_mk_m2_ch6=element_mk_m2_ch6[k];
                  end
                  if(element_mk_m2_ch7[k]>out_temp_mk_m2_ch7)begin
                      out_temp_mk_m2_ch7=element_mk_m2_ch7[k];
                  end
                  if(element_mk_m2_ch8[k]>out_temp_mk_m2_ch8)begin
                      out_temp_mk_m2_ch8=element_mk_m2_ch8[k];
                  end
                  if(element_mk_m2_ch9[k]>out_temp_mk_m2_ch9)begin
                      out_temp_mk_m2_ch9=element_mk_m2_ch9[k];
                  end
                  if(element_mk_m2_ch10[k]>out_temp_mk_m2_ch10)begin
                      out_temp_mk_m2_ch10=element_mk_m2_ch10[k];
                  end
                  if(element_mk_m2_ch11[k]>out_temp_mk_m2_ch11)begin
                      out_temp_mk_m2_ch11=element_mk_m2_ch11[k];
                  end
                  if(element_mk_m2_ch12[k]>out_temp_mk_m2_ch12)begin
                      out_temp_mk_m2_ch12=element_mk_m2_ch12[k];
                  end
                  if(element_mk_m2_ch13[k]>out_temp_mk_m2_ch13)begin
                      out_temp_mk_m2_ch13=element_mk_m2_ch13[k];
                  end
                  if(element_mk_m2_ch14[k]>out_temp_mk_m2_ch14)begin
                      out_temp_mk_m2_ch14=element_mk_m2_ch14[k];
                  end
                  if(element_mk_m2_ch15[k]>out_temp_mk_m2_ch15)begin
                      out_temp_mk_m2_ch15=element_mk_m2_ch15[k];
                  end
                  if(element_mk_m2_ch16[k]>out_temp_mk_m2_ch16)begin
                      out_temp_mk_m2_ch16=element_mk_m2_ch16[k];
                  end
              end
              result_temp_mk_m2_ch1=out_temp_mk_m2_ch1;
              result_temp_mk_m2_ch2=out_temp_mk_m2_ch2;
              result_temp_mk_m2_ch3=out_temp_mk_m2_ch3;
              result_temp_mk_m2_ch4=out_temp_mk_m2_ch4;
              result_temp_mk_m2_ch5=out_temp_mk_m2_ch5;
              result_temp_mk_m2_ch6=out_temp_mk_m2_ch6;
              result_temp_mk_m2_ch7=out_temp_mk_m2_ch7;
              result_temp_mk_m2_ch8=out_temp_mk_m2_ch8;
              result_temp_mk_m2_ch9=out_temp_mk_m2_ch9;
              result_temp_mk_m2_ch10=out_temp_mk_m2_ch10;
              result_temp_mk_m2_ch11=out_temp_mk_m2_ch11;
              result_temp_mk_m2_ch12=out_temp_mk_m2_ch12;
              result_temp_mk_m2_ch13=out_temp_mk_m2_ch13;
              result_temp_mk_m2_ch14=out_temp_mk_m2_ch14;
              result_temp_mk_m2_ch15=out_temp_mk_m2_ch15;
              result_temp_mk_m2_ch16=out_temp_mk_m2_ch16;
              done_mk_m2=1'b1;
          end
      end
          ////////////////////////////////start of dense 1 always block /////////////////////////////////
      if(rst_d1)begin
          count_ld_d1=0;
          done_load_d1=0;
      end
      else if(en_d1 && !done_load_d1) begin
          if(count_ld_d1<10)begin
              addr_d1 = OFFSET_D1_B+count_ld_d1;
              count_ld_d1 = count_ld_d1+1;
              done_load_d1=0;
          end
          else if(count_ld_d1==10)begin
              done_load_d1=0;
              count_ld_d1 = count_ld_d1+1;
          end
          else begin
              count_ld_d1=11'hX;
              done_load_d1=1'b1;
          end
      end
      if(rst_d1==0 && done_load_d1)begin
          case(state_d1)
              0:begin
                  if(en_d1) state_d1=2; else state_d1=0;
              end
              1:begin
                  if(done_nc_d1) state_d1=0; else state_d1=1;
              end
              2:begin
                  if(!done_nc_d1) state_d1=1; else state_d1=2;
              end
          endcase
      end
          /////////////////////////////// neuron calculation ////////////////////////////////
      if(rst_nc_d1)begin
          count_ld_nc_d1=0;
          done_load_nc_d1=0;
      end
      else if(en_nc_d1 && !done_load_nc_d1) begin
          if(count_ld_nc_d1<10)begin
              addr_d1 = OFFSET_D1_W+count_ld_nc_d1*400+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<20)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-10)*400+25+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<30)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-20)*400+50+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<40)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-30)*400+75+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<50)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-40)*400+100+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<60)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-50)*400+125+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<70)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-60)*400+150+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<80)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-70)*400+175+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<90)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-80)*400+200+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<100)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-90)*400+225+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<110)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-100)*400+250+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<120)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-110)*400+275+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<130)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-120)*400+300+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<140)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-130)*400+325+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<150)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-140)*400+350+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1<160)begin
              addr_d1 = OFFSET_D1_W+(count_ld_nc_d1-150)*400+375+weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1==160)begin
              wr_en_buf_1_m2_ch1=1'b0;
              wr_en_buf_1_m2_ch2=1'b0;
              wr_en_buf_1_m2_ch3=1'b0;
              wr_en_buf_1_m2_ch4=1'b0;
              wr_en_buf_1_m2_ch5=1'b0;
              wr_en_buf_1_m2_ch6=1'b0;
              wr_en_buf_1_m2_ch7=1'b0;
              wr_en_buf_1_m2_ch8=1'b0;
              wr_en_buf_1_m2_ch9=1'b0;
              wr_en_buf_1_m2_ch10=1'b0;
              wr_en_buf_1_m2_ch11=1'b0;
              wr_en_buf_1_m2_ch12=1'b0;
              wr_en_buf_1_m2_ch13=1'b0;
              wr_en_buf_1_m2_ch14=1'b0;
              wr_en_buf_1_m2_ch15=1'b0;
              wr_en_buf_1_m2_ch16=1'b0;
              oe_buf_1_m2_ch1=1'b1;
              oe_buf_1_m2_ch2=1'b1;
              oe_buf_1_m2_ch3=1'b1;
              oe_buf_1_m2_ch4=1'b1;
              oe_buf_1_m2_ch5=1'b1;
              oe_buf_1_m2_ch6=1'b1;
              oe_buf_1_m2_ch7=1'b1;
              oe_buf_1_m2_ch8=1'b1;
              oe_buf_1_m2_ch9=1'b1;
              oe_buf_1_m2_ch10=1'b1;
              oe_buf_1_m2_ch11=1'b1;
              oe_buf_1_m2_ch12=1'b1;
              oe_buf_1_m2_ch13=1'b1;
              oe_buf_1_m2_ch14=1'b1;
              oe_buf_1_m2_ch15=1'b1;
              oe_buf_1_m2_ch16=1'b1;
              addr_buf_1_m2_ch1=weight_select_d1;
              addr_buf_1_m2_ch2=weight_select_d1;
              addr_buf_1_m2_ch3=weight_select_d1;
              addr_buf_1_m2_ch4=weight_select_d1;
              addr_buf_1_m2_ch5=weight_select_d1;
              addr_buf_1_m2_ch6=weight_select_d1;
              addr_buf_1_m2_ch7=weight_select_d1;
              addr_buf_1_m2_ch8=weight_select_d1;
              addr_buf_1_m2_ch9=weight_select_d1;
              addr_buf_1_m2_ch10=weight_select_d1;
              addr_buf_1_m2_ch11=weight_select_d1;
              addr_buf_1_m2_ch12=weight_select_d1;
              addr_buf_1_m2_ch13=weight_select_d1;
              addr_buf_1_m2_ch14=weight_select_d1;
              addr_buf_1_m2_ch15=weight_select_d1;
              addr_buf_1_m2_ch16=weight_select_d1;
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else if(count_ld_nc_d1==161)begin
              count_ld_nc_d1 = count_ld_nc_d1+1;
              done_load_nc_d1=0;
          end
          else begin
              oe=1'b0;
              count_ld_nc_d1=11'hX;
              done_load_nc_d1=1'b1;
          end
      end
      if(rst_nc_d1==0 && done_load_nc_d1) begin
          case(state_nc_d1)
              0:begin
                  if(en_nc_d1 && done_load_nc_d1) state_nc_d1=2; else state_nc_d1=0;
              end
              1:begin
                  if(done_m_nc_d1_ch1 && done_m_nc_d1_ch2 && done_m_nc_d1_ch3 && done_m_nc_d1_ch4 && done_m_nc_d1_ch5 && done_m_nc_d1_ch6 && done_m_nc_d1_ch7 && done_m_nc_d1_ch8 && done_m_nc_d1_ch9 && done_m_nc_d1_ch10 && done_m_nc_d1_ch11 && done_m_nc_d1_ch12 && done_m_nc_d1_ch13 && done_m_nc_d1_ch14 && done_m_nc_d1_ch15 && done_m_nc_d1_ch16) state_nc_d1=0; else state_nc_d1=1;
              end
              2:begin
                  if(!done_m_nc_d1_ch1 && !done_m_nc_d1_ch2 && !done_m_nc_d1_ch3 && !done_m_nc_d1_ch4 && !done_m_nc_d1_ch5 && !done_m_nc_d1_ch6 && !done_m_nc_d1_ch7 && !done_m_nc_d1_ch8 && !done_m_nc_d1_ch9 && !done_m_nc_d1_ch10 && !done_m_nc_d1_ch11 && !done_m_nc_d1_ch12 && !done_m_nc_d1_ch13 && !done_m_nc_d1_ch14 && !done_m_nc_d1_ch15 && !done_m_nc_d1_ch16) state_nc_d1=1; else state_nc_d1=2;
              end
          endcase
      end
          ///////////////////////////////end of dense 1 always block //////////////////////////////
          ////////////////////////////////// soft max /////////////////////////////////
      if(en_sm==1) begin
          values[0]<=fc_out[0:31];
          values[1]<=fc_out[32:63];
          values[2]<=fc_out[64:95];
          values[3]<=fc_out[96:127];
          values[4]<=fc_out[128:159];
          values[5]<=fc_out[160:191];
          values[6]<=fc_out[192:223];
          values[7]<=fc_out[224:255];
          values[8]<=fc_out[256:287];
          values[9]<=fc_out[288:319];
          for(i=0; i<10; i=i+1)begin
              if(values[i]>max)begin
                  max=values[i];
                  temp=i;
              end
          end
      end
      else begin
          i=0;
          temp=0;
      end
  end
          ////////////////////////////////// C1 weight and bias load /////////////////////////////////////
  always@(count_ld_shift_c1)begin
      if(count_ld_shift_c1<11)
          W_shift_c1_ch1={W_shift_c1_ch1[8:71],dout};
      else if(count_ld_shift_c1<20)
          W_shift_c1_ch2={W_shift_c1_ch2[8:71],dout};
      else if(count_ld_shift_c1<29)
          W_shift_c1_ch3={W_shift_c1_ch3[8:71],dout};
      else if(count_ld_shift_c1<38)
          W_shift_c1_ch4={W_shift_c1_ch4[8:71],dout};
      else if(count_ld_shift_c1<47)
          W_shift_c1_ch5={W_shift_c1_ch5[8:71],dout};
      else if(count_ld_shift_c1<56)
          W_shift_c1_ch6={W_shift_c1_ch6[8:71],dout};
      else if(count_ld_shift_c1==56)
          bias_shift_c1_ch1=dout;
      else if(count_ld_shift_c1==57)
          bias_shift_c1_ch2=dout;
      else if(count_ld_shift_c1==58)
          bias_shift_c1_ch3=dout;
      else if(count_ld_shift_c1==59)
          bias_shift_c1_ch4=dout;
      else if(count_ld_shift_c1==60)
          bias_shift_c1_ch5=dout;
      else if(count_ld_shift_c1==61)
          bias_shift_c1_ch6=dout;
  end
      ////////////////////////////////// c1 data load ///////////////////////////////////
  always@(count_ld_conv_c1)begin
      if(count_ld_conv_c1<11)
          X_conv_c1={X_conv_c1[8:71],dout};
  end
      ////////////////////////////////// m1 data load /////////////////////////////////
  always@(count_ld_mk_m1)begin
      if(count_ld_mk_m1<6)begin
          X_mk_m1_ch1={X_mk_m1_ch1[16:63],dout_buf_1_c1_ch1};
          X_mk_m1_ch2={X_mk_m1_ch2[16:63],dout_buf_1_c1_ch2};
          X_mk_m1_ch3={X_mk_m1_ch3[16:63],dout_buf_1_c1_ch3};
          X_mk_m1_ch4={X_mk_m1_ch4[16:63],dout_buf_1_c1_ch4};
          X_mk_m1_ch5={X_mk_m1_ch5[16:63],dout_buf_1_c1_ch5};
          X_mk_m1_ch6={X_mk_m1_ch6[16:63],dout_buf_1_c1_ch6};
      end
  end
      ////////////////////////////////// c2 weight and bias load//////////////////////////////////
  always@(count_ld_shift_c2d)begin
      if(count_ld_shift_c2d<11)
          W_shift_c2d_w11={W_shift_c2d_w11[8:71],dout_c2};
      else if(count_ld_shift_c2d<20)
          W_shift_c2d_w12={W_shift_c2d_w12[8:71],dout_c2};
      else if(count_ld_shift_c2d<29)
          W_shift_c2d_w13={W_shift_c2d_w13[8:71],dout_c2};
      else if(count_ld_shift_c2d<38)
          W_shift_c2d_w14={W_shift_c2d_w14[8:71],dout_c2};
      else if(count_ld_shift_c2d<47)
          W_shift_c2d_w15={W_shift_c2d_w15[8:71],dout_c2};
      else if(count_ld_shift_c2d<56)
          W_shift_c2d_w16={W_shift_c2d_w16[8:71],dout_c2};
      else if(count_ld_shift_c2d<65)
          W_shift_c2d_w21={W_shift_c2d_w21[8:71],dout_c2};
      else if(count_ld_shift_c2d<74)
          W_shift_c2d_w22={W_shift_c2d_w22[8:71],dout_c2};
      else if(count_ld_shift_c2d<83)
          W_shift_c2d_w23={W_shift_c2d_w23[8:71],dout_c2};
      else if(count_ld_shift_c2d<92)
          W_shift_c2d_w24={W_shift_c2d_w24[8:71],dout_c2};
      else if(count_ld_shift_c2d<101)
          W_shift_c2d_w25={W_shift_c2d_w25[8:71],dout_c2};
      else if(count_ld_shift_c2d<110)
          W_shift_c2d_w26={W_shift_c2d_w26[8:71],dout_c2};
      else if(count_ld_shift_c2d<119)
          W_shift_c2d_w31={W_shift_c2d_w31[8:71],dout_c2};
      else if(count_ld_shift_c2d<128)
          W_shift_c2d_w32={W_shift_c2d_w32[8:71],dout_c2};
      else if(count_ld_shift_c2d<137)
          W_shift_c2d_w33={W_shift_c2d_w33[8:71],dout_c2};
      else if(count_ld_shift_c2d<146)
          W_shift_c2d_w34={W_shift_c2d_w34[8:71],dout_c2};
      else if(count_ld_shift_c2d<155)
          W_shift_c2d_w35={W_shift_c2d_w35[8:71],dout_c2};
      else if(count_ld_shift_c2d<164)
          W_shift_c2d_w36={W_shift_c2d_w36[8:71],dout_c2};
      else if(count_ld_shift_c2d<173)
          W_shift_c2d_w41={W_shift_c2d_w41[8:71],dout_c2};
      else if(count_ld_shift_c2d<182)
          W_shift_c2d_w42={W_shift_c2d_w42[8:71],dout_c2};
      else if(count_ld_shift_c2d<191)
          W_shift_c2d_w43={W_shift_c2d_w43[8:71],dout_c2};
      else if(count_ld_shift_c2d<200)
          W_shift_c2d_w44={W_shift_c2d_w44[8:71],dout_c2};
      else if(count_ld_shift_c2d<209)
          W_shift_c2d_w45={W_shift_c2d_w45[8:71],dout_c2};
      else if(count_ld_shift_c2d<218)
          W_shift_c2d_w46={W_shift_c2d_w46[8:71],dout_c2};
      else if(count_ld_shift_c2d<227)
          W_shift_c2d_w51={W_shift_c2d_w51[8:71],dout_c2};
      else if(count_ld_shift_c2d<236)
          W_shift_c2d_w52={W_shift_c2d_w52[8:71],dout_c2};
      else if(count_ld_shift_c2d<245)
          W_shift_c2d_w53={W_shift_c2d_w53[8:71],dout_c2};
      else if(count_ld_shift_c2d<254)
          W_shift_c2d_w54={W_shift_c2d_w54[8:71],dout_c2};
      else if(count_ld_shift_c2d<263)
          W_shift_c2d_w55={W_shift_c2d_w55[8:71],dout_c2};
      else if(count_ld_shift_c2d<272)
          W_shift_c2d_w56={W_shift_c2d_w56[8:71],dout_c2};
      else if(count_ld_shift_c2d<281)
          W_shift_c2d_w61={W_shift_c2d_w61[8:71],dout_c2};
      else if(count_ld_shift_c2d<290)
          W_shift_c2d_w62={W_shift_c2d_w62[8:71],dout_c2};
      else if(count_ld_shift_c2d<299)
          W_shift_c2d_w63={W_shift_c2d_w63[8:71],dout_c2};
      else if(count_ld_shift_c2d<308)
          W_shift_c2d_w64={W_shift_c2d_w64[8:71],dout_c2};
      else if(count_ld_shift_c2d<317)
          W_shift_c2d_w65={W_shift_c2d_w65[8:71],dout_c2};
      else if(count_ld_shift_c2d<326)
          W_shift_c2d_w66={W_shift_c2d_w66[8:71],dout_c2};
      else if(count_ld_shift_c2d<335)
          W_shift_c2d_w71={W_shift_c2d_w71[8:71],dout_c2};
      else if(count_ld_shift_c2d<344)
          W_shift_c2d_w72={W_shift_c2d_w72[8:71],dout_c2};
      else if(count_ld_shift_c2d<353)
          W_shift_c2d_w73={W_shift_c2d_w73[8:71],dout_c2};
      else if(count_ld_shift_c2d<362)
          W_shift_c2d_w74={W_shift_c2d_w74[8:71],dout_c2};
      else if(count_ld_shift_c2d<371)
          W_shift_c2d_w75={W_shift_c2d_w75[8:71],dout_c2};
      else if(count_ld_shift_c2d<380)
          W_shift_c2d_w76={W_shift_c2d_w76[8:71],dout_c2};
      else if(count_ld_shift_c2d<389)
          W_shift_c2d_w81={W_shift_c2d_w81[8:71],dout_c2};
      else if(count_ld_shift_c2d<398)
          W_shift_c2d_w82={W_shift_c2d_w82[8:71],dout_c2};
      else if(count_ld_shift_c2d<407)
          W_shift_c2d_w83={W_shift_c2d_w83[8:71],dout_c2};
      else if(count_ld_shift_c2d<416)
          W_shift_c2d_w84={W_shift_c2d_w84[8:71],dout_c2};
      else if(count_ld_shift_c2d<425)
          W_shift_c2d_w85={W_shift_c2d_w85[8:71],dout_c2};
      else if(count_ld_shift_c2d<434)
          W_shift_c2d_w86={W_shift_c2d_w86[8:71],dout_c2};
      else if(count_ld_shift_c2d<443)
          W_shift_c2d_w91={W_shift_c2d_w91[8:71],dout_c2};
      else if(count_ld_shift_c2d<452)
          W_shift_c2d_w92={W_shift_c2d_w92[8:71],dout_c2};
      else if(count_ld_shift_c2d<461)
          W_shift_c2d_w93={W_shift_c2d_w93[8:71],dout_c2};
      else if(count_ld_shift_c2d<470)
          W_shift_c2d_w94={W_shift_c2d_w94[8:71],dout_c2};
      else if(count_ld_shift_c2d<479)
          W_shift_c2d_w95={W_shift_c2d_w95[8:71],dout_c2};
      else if(count_ld_shift_c2d<488)
          W_shift_c2d_w96={W_shift_c2d_w96[8:71],dout_c2};
      else if(count_ld_shift_c2d<497)
          W_shift_c2d_w101={W_shift_c2d_w101[8:71],dout_c2};
      else if(count_ld_shift_c2d<506)
          W_shift_c2d_w102={W_shift_c2d_w102[8:71],dout_c2};
      else if(count_ld_shift_c2d<515)
          W_shift_c2d_w103={W_shift_c2d_w103[8:71],dout_c2};
      else if(count_ld_shift_c2d<524)
          W_shift_c2d_w104={W_shift_c2d_w104[8:71],dout_c2};
      else if(count_ld_shift_c2d<533)
          W_shift_c2d_w105={W_shift_c2d_w105[8:71],dout_c2};
      else if(count_ld_shift_c2d<542)
          W_shift_c2d_w106={W_shift_c2d_w106[8:71],dout_c2};
      else if(count_ld_shift_c2d<551)
          W_shift_c2d_w111={W_shift_c2d_w111[8:71],dout_c2};
      else if(count_ld_shift_c2d<560)
          W_shift_c2d_w112={W_shift_c2d_w112[8:71],dout_c2};
      else if(count_ld_shift_c2d<569)
          W_shift_c2d_w113={W_shift_c2d_w113[8:71],dout_c2};
      else if(count_ld_shift_c2d<578)
          W_shift_c2d_w114={W_shift_c2d_w114[8:71],dout_c2};
      else if(count_ld_shift_c2d<587)
          W_shift_c2d_w115={W_shift_c2d_w115[8:71],dout_c2};
      else if(count_ld_shift_c2d<596)
          W_shift_c2d_w116={W_shift_c2d_w116[8:71],dout_c2};
      else if(count_ld_shift_c2d<605)
          W_shift_c2d_w121={W_shift_c2d_w121[8:71],dout_c2};
      else if(count_ld_shift_c2d<614)
          W_shift_c2d_w122={W_shift_c2d_w122[8:71],dout_c2};
      else if(count_ld_shift_c2d<623)
          W_shift_c2d_w123={W_shift_c2d_w123[8:71],dout_c2};
      else if(count_ld_shift_c2d<632)
          W_shift_c2d_w124={W_shift_c2d_w124[8:71],dout_c2};
      else if(count_ld_shift_c2d<641)
          W_shift_c2d_w125={W_shift_c2d_w125[8:71],dout_c2};
      else if(count_ld_shift_c2d<650)
          W_shift_c2d_w126={W_shift_c2d_w126[8:71],dout_c2};
      else if(count_ld_shift_c2d<659)
          W_shift_c2d_w131={W_shift_c2d_w131[8:71],dout_c2};
      else if(count_ld_shift_c2d<668)
          W_shift_c2d_w132={W_shift_c2d_w132[8:71],dout_c2};
      else if(count_ld_shift_c2d<677)
          W_shift_c2d_w133={W_shift_c2d_w133[8:71],dout_c2};
      else if(count_ld_shift_c2d<686)
          W_shift_c2d_w134={W_shift_c2d_w134[8:71],dout_c2};
      else if(count_ld_shift_c2d<695)
          W_shift_c2d_w135={W_shift_c2d_w135[8:71],dout_c2};
      else if(count_ld_shift_c2d<704)
          W_shift_c2d_w136={W_shift_c2d_w136[8:71],dout_c2};
      else if(count_ld_shift_c2d<713)
          W_shift_c2d_w141={W_shift_c2d_w141[8:71],dout_c2};
      else if(count_ld_shift_c2d<722)
          W_shift_c2d_w142={W_shift_c2d_w142[8:71],dout_c2};
      else if(count_ld_shift_c2d<731)
          W_shift_c2d_w143={W_shift_c2d_w143[8:71],dout_c2};
      else if(count_ld_shift_c2d<740)
          W_shift_c2d_w144={W_shift_c2d_w144[8:71],dout_c2};
      else if(count_ld_shift_c2d<749)
          W_shift_c2d_w145={W_shift_c2d_w145[8:71],dout_c2};
      else if(count_ld_shift_c2d<758)
          W_shift_c2d_w146={W_shift_c2d_w146[8:71],dout_c2};
      else if(count_ld_shift_c2d<767)
          W_shift_c2d_w151={W_shift_c2d_w151[8:71],dout_c2};
      else if(count_ld_shift_c2d<776)
          W_shift_c2d_w152={W_shift_c2d_w152[8:71],dout_c2};
      else if(count_ld_shift_c2d<785)
          W_shift_c2d_w153={W_shift_c2d_w153[8:71],dout_c2};
      else if(count_ld_shift_c2d<794)
          W_shift_c2d_w154={W_shift_c2d_w154[8:71],dout_c2};
      else if(count_ld_shift_c2d<803)
          W_shift_c2d_w155={W_shift_c2d_w155[8:71],dout_c2};
      else if(count_ld_shift_c2d<812)
          W_shift_c2d_w156={W_shift_c2d_w156[8:71],dout_c2};
      else if(count_ld_shift_c2d<821)
          W_shift_c2d_w161={W_shift_c2d_w161[8:71],dout_c2};
      else if(count_ld_shift_c2d<830)
          W_shift_c2d_w162={W_shift_c2d_w162[8:71],dout_c2};
      else if(count_ld_shift_c2d<839)
          W_shift_c2d_w163={W_shift_c2d_w163[8:71],dout_c2};
      else if(count_ld_shift_c2d<848)
          W_shift_c2d_w164={W_shift_c2d_w164[8:71],dout_c2};
      else if(count_ld_shift_c2d<857)
          W_shift_c2d_w165={W_shift_c2d_w165[8:71],dout_c2};
      else if(count_ld_shift_c2d<866)
          W_shift_c2d_w166={W_shift_c2d_w166[8:71],dout_c2};
      else if(count_ld_shift_c2d==866)
          bias_c2d_ch1=dout_c2;
      else if(count_ld_shift_c2d==867)
          bias_c2d_ch2=dout_c2;
      else if(count_ld_shift_c2d==868)
          bias_c2d_ch3=dout_c2;
      else if(count_ld_shift_c2d==869)
          bias_c2d_ch4=dout_c2;
      else if(count_ld_shift_c2d==870)
          bias_c2d_ch5=dout_c2;
      else if(count_ld_shift_c2d==871)
          bias_c2d_ch6=dout_c2;
      else if(count_ld_shift_c2d==872)
          bias_c2d_ch7=dout_c2;
      else if(count_ld_shift_c2d==873)
          bias_c2d_ch8=dout_c2;
      else if(count_ld_shift_c2d==874)
          bias_c2d_ch9=dout_c2;
      else if(count_ld_shift_c2d==875)
          bias_c2d_ch10=dout_c2;
      else if(count_ld_shift_c2d==876)
          bias_c2d_ch11=dout_c2;
      else if(count_ld_shift_c2d==877)
          bias_c2d_ch12=dout_c2;
      else if(count_ld_shift_c2d==878)
          bias_c2d_ch13=dout_c2;
      else if(count_ld_shift_c2d==879)
          bias_c2d_ch14=dout_c2;
      else if(count_ld_shift_c2d==880)
          bias_c2d_ch15=dout_c2;
      else if(count_ld_shift_c2d==881)
          bias_c2d_ch16=dout_c2;
  end
      ////////////////////////////////c2 data load///////////////////////////////////////
  always@(count_ld_conv_c2)begin
      if(count_ld_conv_c2<11)begin
          X_conv_c2_ch_in1={X_conv_c2_ch_in1[16:143],dout_buf_1_m1_ch1};
          X_conv_c2_ch_in2={X_conv_c2_ch_in2[16:143],dout_buf_1_m1_ch2};
          X_conv_c2_ch_in3={X_conv_c2_ch_in3[16:143],dout_buf_1_m1_ch3};
          X_conv_c2_ch_in4={X_conv_c2_ch_in4[16:143],dout_buf_1_m1_ch4};
          X_conv_c2_ch_in5={X_conv_c2_ch_in5[16:143],dout_buf_1_m1_ch5};
          X_conv_c2_ch_in6={X_conv_c2_ch_in6[16:143],dout_buf_1_m1_ch6};
      end
  end
      /////////////////////////////////// m2 data load ///////////////////////////////////////
  always@(count_ld_mk_m2)begin
      if(count_ld_mk_m2<11)begin
          X_mk_m2_ch1={X_mk_m2_ch1[16:143],dout_buf_1_c2_ch1};
          X_mk_m2_ch2={X_mk_m2_ch2[16:143],dout_buf_1_c2_ch2};
          X_mk_m2_ch3={X_mk_m2_ch3[16:143],dout_buf_1_c2_ch3};
          X_mk_m2_ch4={X_mk_m2_ch4[16:143],dout_buf_1_c2_ch4};
          X_mk_m2_ch5={X_mk_m2_ch5[16:143],dout_buf_1_c2_ch5};
          X_mk_m2_ch6={X_mk_m2_ch6[16:143],dout_buf_1_c2_ch6};
          X_mk_m2_ch7={X_mk_m2_ch7[16:143],dout_buf_1_c2_ch7};
          X_mk_m2_ch8={X_mk_m2_ch8[16:143],dout_buf_1_c2_ch8};
          X_mk_m2_ch9={X_mk_m2_ch9[16:143],dout_buf_1_c2_ch9};
          X_mk_m2_ch10={X_mk_m2_ch10[16:143],dout_buf_1_c2_ch10};
          X_mk_m2_ch11={X_mk_m2_ch11[16:143],dout_buf_1_c2_ch11};
          X_mk_m2_ch12={X_mk_m2_ch12[16:143],dout_buf_1_c2_ch12};
          X_mk_m2_ch13={X_mk_m2_ch13[16:143],dout_buf_1_c2_ch13};
          X_mk_m2_ch14={X_mk_m2_ch14[16:143],dout_buf_1_c2_ch14};
          X_mk_m2_ch15={X_mk_m2_ch15[16:143],dout_buf_1_c2_ch15};
          X_mk_m2_ch16={X_mk_m2_ch16[16:143],dout_buf_1_c2_ch16};
      end
  end
      /////////////////////////////////// d1 weight and bias load /////////////////////////////////////
  always@(count_ld_d1)begin
      if(count_ld_d1<12)
          bias_d1={bias_d1[8:79], dout_d1};
  end
  
  always@(count_ld_nc_d1)begin
      if(count_ld_nc_d1<12)
          w_neuron_nc_d1_ch1={w_neuron_nc_d1_ch1[8:79], dout_d1};
      else if(count_ld_nc_d1<22)
          w_neuron_nc_d1_ch2={w_neuron_nc_d1_ch2[8:79], dout_d1};
      else if(count_ld_nc_d1<32)
          w_neuron_nc_d1_ch3={w_neuron_nc_d1_ch3[8:79], dout_d1};
      else if(count_ld_nc_d1<42)
          w_neuron_nc_d1_ch4={w_neuron_nc_d1_ch4[8:79], dout_d1};
      else if(count_ld_nc_d1<52)
          w_neuron_nc_d1_ch5={w_neuron_nc_d1_ch5[8:79], dout_d1};
      else if(count_ld_nc_d1<62)
          w_neuron_nc_d1_ch6={w_neuron_nc_d1_ch6[8:79], dout_d1};
      else if(count_ld_nc_d1<72)
          w_neuron_nc_d1_ch7={w_neuron_nc_d1_ch7[8:79], dout_d1};
      else if(count_ld_nc_d1<82)
          w_neuron_nc_d1_ch8={w_neuron_nc_d1_ch8[8:79], dout_d1};
      else if(count_ld_nc_d1<92)
          w_neuron_nc_d1_ch9={w_neuron_nc_d1_ch9[8:79], dout_d1};
      else if(count_ld_nc_d1<102)
          w_neuron_nc_d1_ch10={w_neuron_nc_d1_ch10[8:79], dout_d1};
      else if(count_ld_nc_d1<112)
          w_neuron_nc_d1_ch11={w_neuron_nc_d1_ch11[8:79], dout_d1};
      else if(count_ld_nc_d1<122)
          w_neuron_nc_d1_ch12={w_neuron_nc_d1_ch12[8:79], dout_d1};
      else if(count_ld_nc_d1<132)
          w_neuron_nc_d1_ch13={w_neuron_nc_d1_ch13[8:79], dout_d1};
      else if(count_ld_nc_d1<142)
          w_neuron_nc_d1_ch14={w_neuron_nc_d1_ch14[8:79], dout_d1};
      else if(count_ld_nc_d1<152)
          w_neuron_nc_d1_ch15={w_neuron_nc_d1_ch15[8:79], dout_d1};
      else if(count_ld_nc_d1<162)
          w_neuron_nc_d1_ch16={w_neuron_nc_d1_ch16[8:79], dout_d1};
      ////////////////////////////////////// d1 data load /////////////////////////////////////////
      else if(count_ld_nc_d1<163)begin
          X_nc_d1_ch1=dout_buf_1_m2_ch1;
          X_nc_d1_ch2=dout_buf_1_m2_ch2;
          X_nc_d1_ch3=dout_buf_1_m2_ch3;
          X_nc_d1_ch4=dout_buf_1_m2_ch4;
          X_nc_d1_ch5=dout_buf_1_m2_ch5;
          X_nc_d1_ch6=dout_buf_1_m2_ch6;
          X_nc_d1_ch7=dout_buf_1_m2_ch7;
          X_nc_d1_ch8=dout_buf_1_m2_ch8;
          X_nc_d1_ch9=dout_buf_1_m2_ch9;
          X_nc_d1_ch10=dout_buf_1_m2_ch10;
          X_nc_d1_ch11=dout_buf_1_m2_ch11;
          X_nc_d1_ch12=dout_buf_1_m2_ch12;
          X_nc_d1_ch13=dout_buf_1_m2_ch13;
          X_nc_d1_ch14=dout_buf_1_m2_ch14;
          X_nc_d1_ch15=dout_buf_1_m2_ch15;
          X_nc_d1_ch16=dout_buf_1_m2_ch16;
      end
  end
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      /////////////////////////////////////////////////////Convolution Layer 1/////////////////////////////////////////////////
      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  always@(posedge done_shift_c1 or posedge rst_c1)begin
      if(rst_c1)begin
          count_c1=0;
      end
      else begin
          if(count_c1<1)begin
              count_c1=count_c1+1;
          end
          else
              count_c1=2'hX;
      end
  end
      ////////////////////////////////////// C1 //////////////////////////////////////
  always@(*)begin
      case(state_c1)
          0:begin
              en_shift_c1=0;
              rst_shift_c1=1;
          end
          1:begin
              en_shift_c1=1;
              rst_shift_c1=0;
          end
          2:begin
              en_shift_c1=0;
              rst_shift_c1=1;
          end
          default:begin
              en_shift_c1=0;
              rst_shift_c1=1;
          end
      endcase
      if(count_c1==1)begin
          done_c1=1'b1;
      end
      else begin
          done_c1=1'b0;
      end
  end
      //////////////////////////////////shift c1//////////////////////////////////////
  always@(posedge done_conv_c1 or posedge rst_shift_c1)begin
      if(rst_shift_c1)begin
          count_shift_c1=0;
          din_buf_0_c1_ch1=0;
          din_buf_0_c1_ch2=0;
          din_buf_0_c1_ch3=0;
          din_buf_0_c1_ch4=0;
          din_buf_0_c1_ch5=0;
          din_buf_0_c1_ch6=0;
      end
      else begin
          if(count_shift_c1<676)begin
              count_shift_c1=count_shift_c1+1;
              din_buf_0_c1_ch1=result_temp_conv_c1_ch1;
              din_buf_0_c1_ch2=result_temp_conv_c1_ch2;
              din_buf_0_c1_ch3=result_temp_conv_c1_ch3;
              din_buf_0_c1_ch4=result_temp_conv_c1_ch4;
              din_buf_0_c1_ch5=result_temp_conv_c1_ch5;
              din_buf_0_c1_ch6=result_temp_conv_c1_ch6;
          end
          else
              count_shift_c1=10'hX;
      end
  end
  
  always@(*)begin
      case(state_shift_c1)
          0:begin
              en_conv_c1=0;
              rst_conv_c1=1;
              window_select_c1=0;
          end
          1:begin
              en_conv_c1=1;
              rst_conv_c1=0;
              window_select_c1=count_shift_c1;
          end
          2:begin
              en_conv_c1=0;
              rst_conv_c1=1;
              window_select_c1=0;
          end
          default:begin
              en_conv_c1=0;
              rst_conv_c1=1;
              window_select_c1=0;
          end
      endcase
      if(count_shift_c1==676)begin
          done_shift_c1=1'b1;
      end
      else begin
          done_shift_c1=1'b0;
      end
  end
  
  assign mem_x_conv_c1[0]= X_conv_c1[0:7];
  assign mem_x_conv_c1[1]= X_conv_c1[8:15];
  assign mem_x_conv_c1[2]= X_conv_c1[16:23];
  assign mem_x_conv_c1[3]= X_conv_c1[24:31];
  assign mem_x_conv_c1[4]= X_conv_c1[32:39];
  assign mem_x_conv_c1[5]= X_conv_c1[40:47];
  assign mem_x_conv_c1[6]= X_conv_c1[48:55];
  assign mem_x_conv_c1[7]= X_conv_c1[56:63];
  assign mem_x_conv_c1[8]= X_conv_c1[64:71];
  
  assign mem_w_conv_c1_ch1[0]= W_shift_c1_ch1[0:7];
  assign mem_w_conv_c1_ch1[1]= W_shift_c1_ch1[8:15];
  assign mem_w_conv_c1_ch1[2]= W_shift_c1_ch1[16:23];
  assign mem_w_conv_c1_ch1[3]= W_shift_c1_ch1[24:31];
  assign mem_w_conv_c1_ch1[4]= W_shift_c1_ch1[32:39];
  assign mem_w_conv_c1_ch1[5]= W_shift_c1_ch1[40:47];
  assign mem_w_conv_c1_ch1[6]= W_shift_c1_ch1[48:55];
  assign mem_w_conv_c1_ch1[7]= W_shift_c1_ch1[56:63];
  assign mem_w_conv_c1_ch1[8]= W_shift_c1_ch1[64:71];
  
  assign mem_w_conv_c1_ch2[0]= W_shift_c1_ch2[0:7];
  assign mem_w_conv_c1_ch2[1]= W_shift_c1_ch2[8:15];
  assign mem_w_conv_c1_ch2[2]= W_shift_c1_ch2[16:23];
  assign mem_w_conv_c1_ch2[3]= W_shift_c1_ch2[24:31];
  assign mem_w_conv_c1_ch2[4]= W_shift_c1_ch2[32:39];
  assign mem_w_conv_c1_ch2[5]= W_shift_c1_ch2[40:47];
  assign mem_w_conv_c1_ch2[6]= W_shift_c1_ch2[48:55];
  assign mem_w_conv_c1_ch2[7]= W_shift_c1_ch2[56:63];
  assign mem_w_conv_c1_ch2[8]= W_shift_c1_ch2[64:71];
  
  assign mem_w_conv_c1_ch3[0]= W_shift_c1_ch3[0:7];
  assign mem_w_conv_c1_ch3[1]= W_shift_c1_ch3[8:15];
  assign mem_w_conv_c1_ch3[2]= W_shift_c1_ch3[16:23];
  assign mem_w_conv_c1_ch3[3]= W_shift_c1_ch3[24:31];
  assign mem_w_conv_c1_ch3[4]= W_shift_c1_ch3[32:39];
  assign mem_w_conv_c1_ch3[5]= W_shift_c1_ch3[40:47];
  assign mem_w_conv_c1_ch3[6]= W_shift_c1_ch3[48:55];
  assign mem_w_conv_c1_ch3[7]= W_shift_c1_ch3[56:63];
  assign mem_w_conv_c1_ch3[8]= W_shift_c1_ch3[64:71];
  
  assign mem_w_conv_c1_ch4[0]= W_shift_c1_ch4[0:7];
  assign mem_w_conv_c1_ch4[1]= W_shift_c1_ch4[8:15];
  assign mem_w_conv_c1_ch4[2]= W_shift_c1_ch4[16:23];
  assign mem_w_conv_c1_ch4[3]= W_shift_c1_ch4[24:31];
  assign mem_w_conv_c1_ch4[4]= W_shift_c1_ch4[32:39];
  assign mem_w_conv_c1_ch4[5]= W_shift_c1_ch4[40:47];
  assign mem_w_conv_c1_ch4[6]= W_shift_c1_ch4[48:55];
  assign mem_w_conv_c1_ch4[7]= W_shift_c1_ch4[56:63];
  assign mem_w_conv_c1_ch4[8]= W_shift_c1_ch4[64:71];
  
  assign mem_w_conv_c1_ch5[0]= W_shift_c1_ch5[0:7];
  assign mem_w_conv_c1_ch5[1]= W_shift_c1_ch5[8:15];
  assign mem_w_conv_c1_ch5[2]= W_shift_c1_ch5[16:23];
  assign mem_w_conv_c1_ch5[3]= W_shift_c1_ch5[24:31];
  assign mem_w_conv_c1_ch5[4]= W_shift_c1_ch5[32:39];
  assign mem_w_conv_c1_ch5[5]= W_shift_c1_ch5[40:47];
  assign mem_w_conv_c1_ch5[6]= W_shift_c1_ch5[48:55];
  assign mem_w_conv_c1_ch5[7]= W_shift_c1_ch5[56:63];
  assign mem_w_conv_c1_ch5[8]= W_shift_c1_ch5[64:71];
  
  assign mem_w_conv_c1_ch6[0]= W_shift_c1_ch6[0:7];
  assign mem_w_conv_c1_ch6[1]= W_shift_c1_ch6[8:15];
  assign mem_w_conv_c1_ch6[2]= W_shift_c1_ch6[16:23];
  assign mem_w_conv_c1_ch6[3]= W_shift_c1_ch6[24:31];
  assign mem_w_conv_c1_ch6[4]= W_shift_c1_ch6[32:39];
  assign mem_w_conv_c1_ch6[5]= W_shift_c1_ch6[40:47];
  assign mem_w_conv_c1_ch6[6]= W_shift_c1_ch6[48:55];
  assign mem_w_conv_c1_ch6[7]= W_shift_c1_ch6[56:63];
  assign mem_w_conv_c1_ch6[8]= W_shift_c1_ch6[64:71];
  
  always@(posedge (done_m_c1_ch1 && done_m_c1_ch2 && done_m_c1_ch3 && done_m_c1_ch4 && done_m_c1_ch5 && done_m_c1_ch6) or posedge rst_conv_c1)begin
      if(rst_conv_c1)begin
          count_conv_c1 = 0;
          buffer_conv_c1_ch1 = 0;
          buffer_conv_c1_ch2 = 0;
          buffer_conv_c1_ch3 = 0;
          buffer_conv_c1_ch4 = 0;
          buffer_conv_c1_ch5 = 0;
          buffer_conv_c1_ch6 = 0;
      end
      else begin
          if(count_conv_c1<9)begin
              count_conv_c1=count_conv_c1+1;
              buffer_conv_c1_ch1=buffer_conv_c1_ch1+result_temp_m_c1_ch1;
              buffer_conv_c1_ch2=buffer_conv_c1_ch2+result_temp_m_c1_ch2;
              buffer_conv_c1_ch3=buffer_conv_c1_ch3+result_temp_m_c1_ch3;
              buffer_conv_c1_ch4=buffer_conv_c1_ch4+result_temp_m_c1_ch4;
              buffer_conv_c1_ch5=buffer_conv_c1_ch5+result_temp_m_c1_ch5;
              buffer_conv_c1_ch6=buffer_conv_c1_ch6+result_temp_m_c1_ch6;
          end
          else begin
              count_conv_c1=4'hx;
          end
      end
  end
  
  always@(*)begin
      case(state_conv_c1)
          0:begin
              en_m_c1=0;
              rst_m_c1=1;
              x_in_m_c1=0;
              w_in_m_c1_ch1=0;
              w_in_m_c1_ch2=0;
              w_in_m_c1_ch3=0;
              w_in_m_c1_ch4=0;
              w_in_m_c1_ch5=0;
              w_in_m_c1_ch6=0;
          end
          1:begin
              en_m_c1=1;
              rst_m_c1=0;
              x_in_m_c1=mem_x_conv_c1[count_conv_c1];
              w_in_m_c1_ch1=mem_w_conv_c1_ch1[count_conv_c1];
              w_in_m_c1_ch2=mem_w_conv_c1_ch2[count_conv_c1];
              w_in_m_c1_ch3=mem_w_conv_c1_ch3[count_conv_c1];
              w_in_m_c1_ch4=mem_w_conv_c1_ch4[count_conv_c1];
              w_in_m_c1_ch5=mem_w_conv_c1_ch5[count_conv_c1];
              w_in_m_c1_ch6=mem_w_conv_c1_ch6[count_conv_c1];
          end
          2:begin
              en_m_c1=0;
              rst_m_c1=1;
              x_in_m_c1=0;
              w_in_m_c1_ch1=0;
              w_in_m_c1_ch2=0;
              w_in_m_c1_ch3=0;
              w_in_m_c1_ch4=0;
              w_in_m_c1_ch5=0;
              w_in_m_c1_ch6=0;
          end
          default:begin
              en_m_c1=0;
              rst_m_c1=1;
              x_in_m_c1=0;
              w_in_m_c1_ch1=0;
              w_in_m_c1_ch2=0;
              w_in_m_c1_ch3=0;
              w_in_m_c1_ch4=0;
              w_in_m_c1_ch5=0;
              w_in_m_c1_ch6=0;
          end
      endcase
      if(count_conv_c1==9)begin
          result_final_temp_conv_c1_ch1=buffer_conv_c1_ch1+bias_shift_c1_ch1;
          result_final_temp_conv_c1_ch1=result_final_temp_conv_c1_ch1>>>4;
          result_temp_conv_c1_ch1=result_final_temp_conv_c1_ch1[15:0];
          result_temp_conv_c1_ch1 = result_temp_conv_c1_ch1 > THRESHOLD ? result_temp_conv_c1_ch1: 16'b0;
  
          result_final_temp_conv_c1_ch2=buffer_conv_c1_ch2+bias_shift_c1_ch2;
          result_final_temp_conv_c1_ch2=result_final_temp_conv_c1_ch2>>>4;
          result_temp_conv_c1_ch2=result_final_temp_conv_c1_ch2[15:0];
          result_temp_conv_c1_ch2 = result_temp_conv_c1_ch2 > THRESHOLD ? result_temp_conv_c1_ch2: 16'b0;
  
          result_final_temp_conv_c1_ch3=buffer_conv_c1_ch3+bias_shift_c1_ch3;
          result_final_temp_conv_c1_ch3=result_final_temp_conv_c1_ch3>>>4;
          result_temp_conv_c1_ch3=result_final_temp_conv_c1_ch3[15:0];
          result_temp_conv_c1_ch3 = result_temp_conv_c1_ch3 > THRESHOLD ? result_temp_conv_c1_ch3: 16'b0;
  
          result_final_temp_conv_c1_ch4=buffer_conv_c1_ch4+bias_shift_c1_ch4;
          result_final_temp_conv_c1_ch4=result_final_temp_conv_c1_ch4>>>4;
          result_temp_conv_c1_ch4=result_final_temp_conv_c1_ch4[15:0];
          result_temp_conv_c1_ch4 = result_temp_conv_c1_ch4 > THRESHOLD ? result_temp_conv_c1_ch4: 16'b0;
  
          result_final_temp_conv_c1_ch5=buffer_conv_c1_ch5+bias_shift_c1_ch5;
          result_final_temp_conv_c1_ch5=result_final_temp_conv_c1_ch5>>>4;
          result_temp_conv_c1_ch5=result_final_temp_conv_c1_ch5[15:0];
          result_temp_conv_c1_ch5 = result_temp_conv_c1_ch5 > THRESHOLD ? result_temp_conv_c1_ch5: 16'b0;
  
          result_final_temp_conv_c1_ch6=buffer_conv_c1_ch6+bias_shift_c1_ch6;
          result_final_temp_conv_c1_ch6=result_final_temp_conv_c1_ch6>>>4;
          result_temp_conv_c1_ch6=result_final_temp_conv_c1_ch6[15:0];
          result_temp_conv_c1_ch6 = result_temp_conv_c1_ch6 > THRESHOLD ? result_temp_conv_c1_ch6: 16'b0;
  
          done_conv_c1=1'b1;
      end
      else begin
          result_temp_conv_c1_ch1=0;
          result_temp_conv_c1_ch2=0;
          result_temp_conv_c1_ch3=0;
          result_temp_conv_c1_ch4=0;
          result_temp_conv_c1_ch5=0;
          result_temp_conv_c1_ch6=0;
          done_conv_c1=1'b0;
      end
  end
  
  element_multiplier_c1#(BITWIDTH_IN,BITWIDTH_W,BITWIDTH_C1) multiply_ch1(
  .clk(clk),
  .in_ready(en_m_c1),
  .rst(rst_m_c1),
  .X_element(x_in_m_c1),
  .W_element(w_in_m_c1_ch1),
  .Z_element(result_temp_m_c1_ch1),
  .done(done_m_c1_ch1)
  );
  
  element_multiplier_c1#(BITWIDTH_IN,BITWIDTH_W,BITWIDTH_C1) multiply_ch2(
  .clk(clk),
  .in_ready(en_m_c1),
  .rst(rst_m_c1),
  .X_element(x_in_m_c1),
  .W_element(w_in_m_c1_ch2),
  .Z_element(result_temp_m_c1_ch2),
  .done(done_m_c1_ch2)
  );
  
  element_multiplier_c1#(BITWIDTH_IN,BITWIDTH_W,BITWIDTH_C1) multiply_ch3(
  .clk(clk),
  .in_ready(en_m_c1),
  .rst(rst_m_c1),
  .X_element(x_in_m_c1),
  .W_element(w_in_m_c1_ch3),
  .Z_element(result_temp_m_c1_ch3),
  .done(done_m_c1_ch3)
  );
  
  element_multiplier_c1#(BITWIDTH_IN,BITWIDTH_W,BITWIDTH_C1) multiply_ch4(
  .clk(clk),
  .in_ready(en_m_c1),
  .rst(rst_m_c1),
  .X_element(x_in_m_c1),
  .W_element(w_in_m_c1_ch4),
  .Z_element(result_temp_m_c1_ch4),
  .done(done_m_c1_ch4)
  );
  
  element_multiplier_c1#(BITWIDTH_IN,BITWIDTH_W,BITWIDTH_C1) multiply_ch5(
  .clk(clk),
  .in_ready(en_m_c1),
  .rst(rst_m_c1),
  .X_element(x_in_m_c1),
  .W_element(w_in_m_c1_ch5),
  .Z_element(result_temp_m_c1_ch5),
  .done(done_m_c1_ch5)
  );
  
  element_multiplier_c1#(BITWIDTH_IN,BITWIDTH_W,BITWIDTH_C1) multiply_ch6(
  .clk(clk),
  .in_ready(en_m_c1),
  .rst(rst_m_c1),
  .X_element(x_in_m_c1),
  .W_element(w_in_m_c1_ch6),
  .Z_element(result_temp_m_c1_ch6),
  .done(done_m_c1_ch6)
  );
  
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //////////////////////////////////////////////////////Maxpool Layer 1///////////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  assign start_addr_mk_m1 = window_select_m1*2+(window_select_m1/13)*26;
  
      ////////////////////////maxpool kernel/////////////////////////
  assign element_mk_m1_ch1[0]=X_mk_m1_ch1[0:15];
  assign element_mk_m1_ch1[1]=X_mk_m1_ch1[16:31];
  assign element_mk_m1_ch1[2]=X_mk_m1_ch1[32:47];
  assign element_mk_m1_ch1[3]=X_mk_m1_ch1[48:63];
  
  assign element_mk_m1_ch2[0]=X_mk_m1_ch2[0:15];
  assign element_mk_m1_ch2[1]=X_mk_m1_ch2[16:31];
  assign element_mk_m1_ch2[2]=X_mk_m1_ch2[32:47];
  assign element_mk_m1_ch2[3]=X_mk_m1_ch2[48:63];
  
  assign element_mk_m1_ch3[0]=X_mk_m1_ch3[0:15];
  assign element_mk_m1_ch3[1]=X_mk_m1_ch3[16:31];
  assign element_mk_m1_ch3[2]=X_mk_m1_ch3[32:47];
  assign element_mk_m1_ch3[3]=X_mk_m1_ch3[48:63];
  
  assign element_mk_m1_ch4[0]=X_mk_m1_ch4[0:15];
  assign element_mk_m1_ch4[1]=X_mk_m1_ch4[16:31];
  assign element_mk_m1_ch4[2]=X_mk_m1_ch4[32:47];
  assign element_mk_m1_ch4[3]=X_mk_m1_ch4[48:63];
  
  assign element_mk_m1_ch5[0]=X_mk_m1_ch5[0:15];
  assign element_mk_m1_ch5[1]=X_mk_m1_ch5[16:31];
  assign element_mk_m1_ch5[2]=X_mk_m1_ch5[32:47];
  assign element_mk_m1_ch5[3]=X_mk_m1_ch5[48:63];
  
  assign element_mk_m1_ch6[0]=X_mk_m1_ch6[0:15];
  assign element_mk_m1_ch6[1]=X_mk_m1_ch6[16:31];
  assign element_mk_m1_ch6[2]=X_mk_m1_ch6[32:47];
  assign element_mk_m1_ch6[3]=X_mk_m1_ch6[48:63];
  
  always@(posedge done_shift_m1 or posedge rst_m1)begin
      if(rst_m1)begin
          count_m1=0;
      end
      else begin
          if(count_m1<1)begin
              count_m1=count_m1+1;
          end
          else
              count_m1=2'hX;
      end
  end
  
  always@(*)begin
       case(state_m1)
          0:begin
              en_shift_m1=0;
              rst_shift_m1=1;
          end
          1:begin
              en_shift_m1=1;
              rst_shift_m1=0;
          end
          2:begin
              en_shift_m1=0;
              rst_shift_m1=1;
          end
          3:begin
              en_shift_m1=0;
              rst_shift_m1=0;
          end
          default:begin
              en_shift_m1=0;
              rst_shift_m1=1;
          end
      endcase
      if(count_m1==1)begin
          done_m1=1'b1;
      end
      else begin
          done_m1=1'b0;
      end
  end
  
      ///////////////////////////////////shift m1/////////////////////////////////////
  always@(posedge done_mk_m1_ch1 or posedge rst_shift_m1)begin
      if(rst_shift_m1)begin
          count_shift_m1=0;
          din_buf_0_m1_ch1=0;
          din_buf_0_m1_ch2=0;
          din_buf_0_m1_ch3=0;
          din_buf_0_m1_ch4=0;
          din_buf_0_m1_ch5=0;
          din_buf_0_m1_ch6=0;
          n=1;
      end
      else begin
          if(count_shift_m1<169)begin
              if(count_shift_m1==n*13-1 && !done_mk_m1_row_ch1 && !done_mk_m1_row_ch2 && !done_mk_m1_row_ch3 && !done_mk_m1_row_ch4 && !done_mk_m1_row_ch5 && !done_mk_m1_row_ch6)begin
                  n=n+1;
                  done_mk_m1_row_ch1=1'b1;
                  done_mk_m1_row_ch2=1'b1;
                  done_mk_m1_row_ch3=1'b1;
                  done_mk_m1_row_ch4=1'b1;
                  done_mk_m1_row_ch5=1'b1;
                  done_mk_m1_row_ch6=1'b1;
              end
              else begin
                  count_shift_m1=count_shift_m1+1;
                  din_buf_0_m1_ch1=result_temp_mk_m1_ch1;
                  din_buf_0_m1_ch2=result_temp_mk_m1_ch2;
                  din_buf_0_m1_ch3=result_temp_mk_m1_ch3;
                  din_buf_0_m1_ch4=result_temp_mk_m1_ch4;
                  din_buf_0_m1_ch5=result_temp_mk_m1_ch5;
                  din_buf_0_m1_ch6=result_temp_mk_m1_ch6;
              end
          end
          else
              count_shift_m1=8'hX;
      end
  end
  
  always@(*)begin
      case(state_shift_m1)
          0:begin
              en_mk_m1=0;
              rst_mk_m1=1;
              window_select_m1=0;
          end
          1:begin
              en_mk_m1=1;
              rst_mk_m1=0;
              window_select_m1=count_shift_m1;
          end
          2:begin
              en_mk_m1=0;
              rst_mk_m1=1;
              window_select_m1=0;
          end
          default:begin
              en_mk_m1=0;
              rst_mk_m1=1;
              window_select_m1=0;
          end
      endcase
      if(count_shift_m1==169)begin
          done_shift_m1=1'b1;
      end
      else begin
          done_shift_m1=1'b0;
      end
  end
  
      ////////////////////////////////////// maxpool kernel /////////////////////////////////////////
  always@(posedge rst_mk_m1)begin
      done_mk_m1_row_ch1=0;
      done_mk_m1_row_ch2=0;
      done_mk_m1_row_ch3=0;
      done_mk_m1_row_ch4=0;
      done_mk_m1_row_ch5=0;
      done_mk_m1_row_ch6=0;
  end
  
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
      //////////////////////////////////////////////////////Convolution Layer 2///////////////////////////////////////////////////
      ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
      ///////////////////////////////////////////////////////  c2d  ///////////////////////////////////////////////////////
  always@(posedge done_shift_c2d or posedge rst_c2)begin
      if(rst_c2)begin
          count_c2d=0;
      end
      else begin
          if(count_c2d<1)begin
              count_c2d=count_c2d+1;
          end
          else
              count_c2d=2'hX;
      end
  end
  
  always@(*)begin
      case(state_c2d)
          0:begin
              en_shift_c2d=0;
              rst_shift_c2d=1;
          end
          1:begin
              en_shift_c2d=1;
              rst_shift_c2d=0;
          end
          2:begin
              en_shift_c2d=0;
              rst_shift_c2d=1;
          end
          3:begin
              en_shift_c2d=0;
              rst_shift_c2d=0;
          end
          default:begin
              en_shift_c2d=0;
              rst_shift_c2d=1;
          end
      endcase
      if(count_c2d==1)
          done_c2=1'b1;
      else 
          done_c2=1'b0;
  end
  
      ////////////////////////////////////// adder function /////////////////////////////////////////
  function signed [0:15] adder;
      input signed [0:15]B;
      reg signed [0:31] add_temp;
      input signed [0:15]Z1,Z2,Z3,Z4,Z5,Z6;
      input signed [0:15] THRESHOLD;
      begin
          add_temp = B+Z1+Z2+Z3+Z4+Z5+Z6;
          if(add_temp>32767 || add_temp<-32768)begin
              if(add_temp>32767)
                  adder = 16'b0111111111111111;
              else
                  adder = 16'b0000000000000000;//-ve values are made 0 as relu is integrated with this module
          end
          else begin
              adder = add_temp[16:31];
          end
          adder = adder > THRESHOLD ? adder: 16'd0;
      end
  endfunction
  
      //////////////////////////////////// shift window c2 ////////////////////////////////////////
  always@(posedge done_ck_c2d or posedge rst_shift_c2d)begin
      if(rst_shift_c2d)begin
          count_shift_c2d=0;
          din_buf_0_c2_ch1=0;
          din_buf_0_c2_ch2=0;
          din_buf_0_c2_ch3=0;
          din_buf_0_c2_ch4=0;
          din_buf_0_c2_ch5=0;
          din_buf_0_c2_ch6=0;
          din_buf_0_c2_ch7=0;
          din_buf_0_c2_ch8=0;
          din_buf_0_c2_ch9=0;
          din_buf_0_c2_ch10=0;
          din_buf_0_c2_ch11=0;
          din_buf_0_c2_ch12=0;
          din_buf_0_c2_ch13=0;
          din_buf_0_c2_ch14=0;
          din_buf_0_c2_ch15=0;
          din_buf_0_c2_ch16=0;
          m=1;
      end
      else begin
          if(count_shift_c2d<121)begin
              if(count_shift_c2d==m*11-1 && !done_ck_c2d_row_ch1 && !done_ck_c2d_row_ch2 && !done_ck_c2d_row_ch3 && !done_ck_c2d_row_ch4 && !done_ck_c2d_row_ch5 && !done_ck_c2d_row_ch6 && !done_ck_c2d_row_ch7 && !done_ck_c2d_row_ch8 && !done_ck_c2d_row_ch9 && !done_ck_c2d_row_ch10 && !done_ck_c2d_row_ch11 && !done_ck_c2d_row_ch12 && !done_ck_c2d_row_ch13 && !done_ck_c2d_row_ch14 && !done_ck_c2d_row_ch15 && !done_ck_c2d_row_ch16)begin
                  m=m+1;
                  done_ck_c2d_row_ch1=1'b1;
                  done_ck_c2d_row_ch2=1'b1;
                  done_ck_c2d_row_ch3=1'b1;
                  done_ck_c2d_row_ch4=1'b1;
                  done_ck_c2d_row_ch5=1'b1;
                  done_ck_c2d_row_ch6=1'b1;
                  done_ck_c2d_row_ch7=1'b1;
                  done_ck_c2d_row_ch8=1'b1;
                  done_ck_c2d_row_ch9=1'b1;
                  done_ck_c2d_row_ch10=1'b1;
                  done_ck_c2d_row_ch11=1'b1;
                  done_ck_c2d_row_ch12=1'b1;
                  done_ck_c2d_row_ch13=1'b1;
                  done_ck_c2d_row_ch14=1'b1;
                  done_ck_c2d_row_ch15=1'b1;
                  done_ck_c2d_row_ch16=1'b1;
              end
              else begin
                  count_shift_c2d=count_shift_c2d+1;
                  din_buf_0_c2_ch1=adder(bias_new_c2d_ch1,result_temp_ck_c2d_in1_w11, result_temp_ck_c2d_in2_w12, result_temp_ck_c2d_in3_w13, result_temp_ck_c2d_in4_w14, result_temp_ck_c2d_in5_w15, result_temp_ck_c2d_in6_w16, THRESHOLD);
                  din_buf_0_c2_ch2=adder(bias_new_c2d_ch2,result_temp_ck_c2d_in1_w21, result_temp_ck_c2d_in2_w22, result_temp_ck_c2d_in3_w23, result_temp_ck_c2d_in4_w24, result_temp_ck_c2d_in5_w25, result_temp_ck_c2d_in6_w26, THRESHOLD);
                  din_buf_0_c2_ch3=adder(bias_new_c2d_ch3,result_temp_ck_c2d_in1_w31, result_temp_ck_c2d_in2_w32, result_temp_ck_c2d_in3_w33, result_temp_ck_c2d_in4_w34, result_temp_ck_c2d_in5_w35, result_temp_ck_c2d_in6_w36, THRESHOLD);
                  din_buf_0_c2_ch4=adder(bias_new_c2d_ch4,result_temp_ck_c2d_in1_w41, result_temp_ck_c2d_in2_w42, result_temp_ck_c2d_in3_w43, result_temp_ck_c2d_in4_w44, result_temp_ck_c2d_in5_w45, result_temp_ck_c2d_in6_w46, THRESHOLD);
                  din_buf_0_c2_ch5=adder(bias_new_c2d_ch5,result_temp_ck_c2d_in1_w51, result_temp_ck_c2d_in2_w52, result_temp_ck_c2d_in3_w53, result_temp_ck_c2d_in4_w54, result_temp_ck_c2d_in5_w55, result_temp_ck_c2d_in6_w56, THRESHOLD);
                  din_buf_0_c2_ch6=adder(bias_new_c2d_ch6,result_temp_ck_c2d_in1_w61, result_temp_ck_c2d_in2_w62, result_temp_ck_c2d_in3_w63, result_temp_ck_c2d_in4_w64, result_temp_ck_c2d_in5_w65, result_temp_ck_c2d_in6_w66, THRESHOLD);
                  din_buf_0_c2_ch7=adder(bias_new_c2d_ch7,result_temp_ck_c2d_in1_w71, result_temp_ck_c2d_in2_w72, result_temp_ck_c2d_in3_w73, result_temp_ck_c2d_in4_w74, result_temp_ck_c2d_in5_w75, result_temp_ck_c2d_in6_w76, THRESHOLD);
                  din_buf_0_c2_ch8=adder(bias_new_c2d_ch8,result_temp_ck_c2d_in1_w81, result_temp_ck_c2d_in2_w82, result_temp_ck_c2d_in3_w83, result_temp_ck_c2d_in4_w84, result_temp_ck_c2d_in5_w85, result_temp_ck_c2d_in6_w86, THRESHOLD);
                  din_buf_0_c2_ch9=adder(bias_new_c2d_ch9,result_temp_ck_c2d_in1_w91, result_temp_ck_c2d_in2_w92, result_temp_ck_c2d_in3_w93, result_temp_ck_c2d_in4_w94, result_temp_ck_c2d_in5_w95, result_temp_ck_c2d_in6_w96, THRESHOLD);
                  din_buf_0_c2_ch10=adder(bias_new_c2d_ch10,result_temp_ck_c2d_in1_w101, result_temp_ck_c2d_in2_w102, result_temp_ck_c2d_in3_w103, result_temp_ck_c2d_in4_w104, result_temp_ck_c2d_in5_w105, result_temp_ck_c2d_in6_w106, THRESHOLD);
                  din_buf_0_c2_ch11=adder(bias_new_c2d_ch11,result_temp_ck_c2d_in1_w111, result_temp_ck_c2d_in2_w112, result_temp_ck_c2d_in3_w113, result_temp_ck_c2d_in4_w114, result_temp_ck_c2d_in5_w115, result_temp_ck_c2d_in6_w116, THRESHOLD);
                  din_buf_0_c2_ch12=adder(bias_new_c2d_ch12,result_temp_ck_c2d_in1_w121, result_temp_ck_c2d_in2_w122, result_temp_ck_c2d_in3_w123, result_temp_ck_c2d_in4_w124, result_temp_ck_c2d_in5_w125, result_temp_ck_c2d_in6_w126, THRESHOLD);
                  din_buf_0_c2_ch13=adder(bias_new_c2d_ch13,result_temp_ck_c2d_in1_w131, result_temp_ck_c2d_in2_w132, result_temp_ck_c2d_in3_w133, result_temp_ck_c2d_in4_w134, result_temp_ck_c2d_in5_w135, result_temp_ck_c2d_in6_w136, THRESHOLD);
                  din_buf_0_c2_ch14=adder(bias_new_c2d_ch14,result_temp_ck_c2d_in1_w141, result_temp_ck_c2d_in2_w142, result_temp_ck_c2d_in3_w143, result_temp_ck_c2d_in4_w144, result_temp_ck_c2d_in5_w145, result_temp_ck_c2d_in6_w146, THRESHOLD);
                  din_buf_0_c2_ch15=adder(bias_new_c2d_ch15,result_temp_ck_c2d_in1_w151, result_temp_ck_c2d_in2_w152, result_temp_ck_c2d_in3_w153, result_temp_ck_c2d_in4_w154, result_temp_ck_c2d_in5_w155, result_temp_ck_c2d_in6_w156, THRESHOLD);
                  din_buf_0_c2_ch16=adder(bias_new_c2d_ch16,result_temp_ck_c2d_in1_w161, result_temp_ck_c2d_in2_w162, result_temp_ck_c2d_in3_w163, result_temp_ck_c2d_in4_w164, result_temp_ck_c2d_in5_w165, result_temp_ck_c2d_in6_w166, THRESHOLD);
              end
          end
          else
              count_shift_c2d=7'hX;
      end
  end
  
  always@(*)begin
      case(state_shift_c2d)
          0:begin
              en_ck_c2d=0;
              rst_ck_c2d=1;
              window_select_c2=0;
          end
          1:begin
              en_ck_c2d=1;
              rst_ck_c2d=0;
              window_select_c2=count_shift_c2d;
          end
          2:begin
              en_ck_c2d=0;
              rst_ck_c2d=1;
              window_select_c2=0;
          end
          default:begin
              en_ck_c2d=0;
              rst_ck_c2d=1;
              window_select_c2=0;
          end
      endcase
      if(count_shift_c2d==121)
          done_shift_c2d=1'b1;
      else
          done_shift_c2d=1'b0;
  end
  
  always@(posedge rst_ck_c2d)begin
      done_ck_c2d_row_ch1=0;
      done_ck_c2d_row_ch2=0;
      done_ck_c2d_row_ch3=0;
      done_ck_c2d_row_ch4=0;
      done_ck_c2d_row_ch5=0;
      done_ck_c2d_row_ch6=0;
      done_ck_c2d_row_ch7=0;
      done_ck_c2d_row_ch8=0;
      done_ck_c2d_row_ch9=0;
      done_ck_c2d_row_ch10=0;
      done_ck_c2d_row_ch11=0;
      done_ck_c2d_row_ch12=0;
      done_ck_c2d_row_ch13=0;
      done_ck_c2d_row_ch14=0;
      done_ck_c2d_row_ch15=0;
      done_ck_c2d_row_ch16=0;
  end
  
  assign mem_x_ck_c2d_ch_in1[0]= X_conv_c2_ch_in1[0:15];
  assign mem_x_ck_c2d_ch_in1[1]= X_conv_c2_ch_in1[16:31];
  assign mem_x_ck_c2d_ch_in1[2]= X_conv_c2_ch_in1[32:47];
  assign mem_x_ck_c2d_ch_in1[3]= X_conv_c2_ch_in1[48:63];
  assign mem_x_ck_c2d_ch_in1[4]= X_conv_c2_ch_in1[64:79];
  assign mem_x_ck_c2d_ch_in1[5]= X_conv_c2_ch_in1[80:95];
  assign mem_x_ck_c2d_ch_in1[6]= X_conv_c2_ch_in1[96:111];
  assign mem_x_ck_c2d_ch_in1[7]= X_conv_c2_ch_in1[112:127];
  assign mem_x_ck_c2d_ch_in1[8]= X_conv_c2_ch_in1[128:143];
  
  assign mem_x_ck_c2d_ch_in2[0]= X_conv_c2_ch_in2[0:15];
  assign mem_x_ck_c2d_ch_in2[1]= X_conv_c2_ch_in2[16:31];
  assign mem_x_ck_c2d_ch_in2[2]= X_conv_c2_ch_in2[32:47];
  assign mem_x_ck_c2d_ch_in2[3]= X_conv_c2_ch_in2[48:63];
  assign mem_x_ck_c2d_ch_in2[4]= X_conv_c2_ch_in2[64:79];
  assign mem_x_ck_c2d_ch_in2[5]= X_conv_c2_ch_in2[80:95];
  assign mem_x_ck_c2d_ch_in2[6]= X_conv_c2_ch_in2[96:111];
  assign mem_x_ck_c2d_ch_in2[7]= X_conv_c2_ch_in2[112:127];
  assign mem_x_ck_c2d_ch_in2[8]= X_conv_c2_ch_in2[128:143];
  
  assign mem_x_ck_c2d_ch_in3[0]= X_conv_c2_ch_in3[0:15];
  assign mem_x_ck_c2d_ch_in3[1]= X_conv_c2_ch_in3[16:31];
  assign mem_x_ck_c2d_ch_in3[2]= X_conv_c2_ch_in3[32:47];
  assign mem_x_ck_c2d_ch_in3[3]= X_conv_c2_ch_in3[48:63];
  assign mem_x_ck_c2d_ch_in3[4]= X_conv_c2_ch_in3[64:79];
  assign mem_x_ck_c2d_ch_in3[5]= X_conv_c2_ch_in3[80:95];
  assign mem_x_ck_c2d_ch_in3[6]= X_conv_c2_ch_in3[96:111];
  assign mem_x_ck_c2d_ch_in3[7]= X_conv_c2_ch_in3[112:127];
  assign mem_x_ck_c2d_ch_in3[8]= X_conv_c2_ch_in3[128:143];
  
  assign mem_x_ck_c2d_ch_in4[0]= X_conv_c2_ch_in4[0:15];
  assign mem_x_ck_c2d_ch_in4[1]= X_conv_c2_ch_in4[16:31];
  assign mem_x_ck_c2d_ch_in4[2]= X_conv_c2_ch_in4[32:47];
  assign mem_x_ck_c2d_ch_in4[3]= X_conv_c2_ch_in4[48:63];
  assign mem_x_ck_c2d_ch_in4[4]= X_conv_c2_ch_in4[64:79];
  assign mem_x_ck_c2d_ch_in4[5]= X_conv_c2_ch_in4[80:95];
  assign mem_x_ck_c2d_ch_in4[6]= X_conv_c2_ch_in4[96:111];
  assign mem_x_ck_c2d_ch_in4[7]= X_conv_c2_ch_in4[112:127];
  assign mem_x_ck_c2d_ch_in4[8]= X_conv_c2_ch_in4[128:143];
  
  assign mem_x_ck_c2d_ch_in5[0]= X_conv_c2_ch_in5[0:15];
  assign mem_x_ck_c2d_ch_in5[1]= X_conv_c2_ch_in5[16:31];
  assign mem_x_ck_c2d_ch_in5[2]= X_conv_c2_ch_in5[32:47];
  assign mem_x_ck_c2d_ch_in5[3]= X_conv_c2_ch_in5[48:63];
  assign mem_x_ck_c2d_ch_in5[4]= X_conv_c2_ch_in5[64:79];
  assign mem_x_ck_c2d_ch_in5[5]= X_conv_c2_ch_in5[80:95];
  assign mem_x_ck_c2d_ch_in5[6]= X_conv_c2_ch_in5[96:111];
  assign mem_x_ck_c2d_ch_in5[7]= X_conv_c2_ch_in5[112:127];
  assign mem_x_ck_c2d_ch_in5[8]= X_conv_c2_ch_in5[128:143];
  
  assign mem_x_ck_c2d_ch_in6[0]= X_conv_c2_ch_in6[0:15];
  assign mem_x_ck_c2d_ch_in6[1]= X_conv_c2_ch_in6[16:31];
  assign mem_x_ck_c2d_ch_in6[2]= X_conv_c2_ch_in6[32:47];
  assign mem_x_ck_c2d_ch_in6[3]= X_conv_c2_ch_in6[48:63];
  assign mem_x_ck_c2d_ch_in6[4]= X_conv_c2_ch_in6[64:79];
  assign mem_x_ck_c2d_ch_in6[5]= X_conv_c2_ch_in6[80:95];
  assign mem_x_ck_c2d_ch_in6[6]= X_conv_c2_ch_in6[96:111];
  assign mem_x_ck_c2d_ch_in6[7]= X_conv_c2_ch_in6[112:127];
  assign mem_x_ck_c2d_ch_in6[8]= X_conv_c2_ch_in6[128:143];
  
  assign mem_w_ck_c2d_w11[0]= W_shift_c2d_w11[0:7];
  assign mem_w_ck_c2d_w11[1]= W_shift_c2d_w11[8:15];
  assign mem_w_ck_c2d_w11[2]= W_shift_c2d_w11[16:23];
  assign mem_w_ck_c2d_w11[3]= W_shift_c2d_w11[24:31];
  assign mem_w_ck_c2d_w11[4]= W_shift_c2d_w11[32:39];
  assign mem_w_ck_c2d_w11[5]= W_shift_c2d_w11[40:47];
  assign mem_w_ck_c2d_w11[6]= W_shift_c2d_w11[48:55];
  assign mem_w_ck_c2d_w11[7]= W_shift_c2d_w11[56:63];
  assign mem_w_ck_c2d_w11[8]= W_shift_c2d_w11[64:71];
  
  assign mem_w_ck_c2d_w12[0]= W_shift_c2d_w12[0:7];
  assign mem_w_ck_c2d_w12[1]= W_shift_c2d_w12[8:15];
  assign mem_w_ck_c2d_w12[2]= W_shift_c2d_w12[16:23];
  assign mem_w_ck_c2d_w12[3]= W_shift_c2d_w12[24:31];
  assign mem_w_ck_c2d_w12[4]= W_shift_c2d_w12[32:39];
  assign mem_w_ck_c2d_w12[5]= W_shift_c2d_w12[40:47];
  assign mem_w_ck_c2d_w12[6]= W_shift_c2d_w12[48:55];
  assign mem_w_ck_c2d_w12[7]= W_shift_c2d_w12[56:63];
  assign mem_w_ck_c2d_w12[8]= W_shift_c2d_w12[64:71];
  
  assign mem_w_ck_c2d_w13[0]= W_shift_c2d_w13[0:7];
  assign mem_w_ck_c2d_w13[1]= W_shift_c2d_w13[8:15];
  assign mem_w_ck_c2d_w13[2]= W_shift_c2d_w13[16:23];
  assign mem_w_ck_c2d_w13[3]= W_shift_c2d_w13[24:31];
  assign mem_w_ck_c2d_w13[4]= W_shift_c2d_w13[32:39];
  assign mem_w_ck_c2d_w13[5]= W_shift_c2d_w13[40:47];
  assign mem_w_ck_c2d_w13[6]= W_shift_c2d_w13[48:55];
  assign mem_w_ck_c2d_w13[7]= W_shift_c2d_w13[56:63];
  assign mem_w_ck_c2d_w13[8]= W_shift_c2d_w13[64:71];
  
  assign mem_w_ck_c2d_w14[0]= W_shift_c2d_w14[0:7];
  assign mem_w_ck_c2d_w14[1]= W_shift_c2d_w14[8:15];
  assign mem_w_ck_c2d_w14[2]= W_shift_c2d_w14[16:23];
  assign mem_w_ck_c2d_w14[3]= W_shift_c2d_w14[24:31];
  assign mem_w_ck_c2d_w14[4]= W_shift_c2d_w14[32:39];
  assign mem_w_ck_c2d_w14[5]= W_shift_c2d_w14[40:47];
  assign mem_w_ck_c2d_w14[6]= W_shift_c2d_w14[48:55];
  assign mem_w_ck_c2d_w14[7]= W_shift_c2d_w14[56:63];
  assign mem_w_ck_c2d_w14[8]= W_shift_c2d_w14[64:71];
  
  assign mem_w_ck_c2d_w15[0]= W_shift_c2d_w15[0:7];
  assign mem_w_ck_c2d_w15[1]= W_shift_c2d_w15[8:15];
  assign mem_w_ck_c2d_w15[2]= W_shift_c2d_w15[16:23];
  assign mem_w_ck_c2d_w15[3]= W_shift_c2d_w15[24:31];
  assign mem_w_ck_c2d_w15[4]= W_shift_c2d_w15[32:39];
  assign mem_w_ck_c2d_w15[5]= W_shift_c2d_w15[40:47];
  assign mem_w_ck_c2d_w15[6]= W_shift_c2d_w15[48:55];
  assign mem_w_ck_c2d_w15[7]= W_shift_c2d_w15[56:63];
  assign mem_w_ck_c2d_w15[8]= W_shift_c2d_w15[64:71];
  
  assign mem_w_ck_c2d_w16[0]= W_shift_c2d_w16[0:7];
  assign mem_w_ck_c2d_w16[1]= W_shift_c2d_w16[8:15];
  assign mem_w_ck_c2d_w16[2]= W_shift_c2d_w16[16:23];
  assign mem_w_ck_c2d_w16[3]= W_shift_c2d_w16[24:31];
  assign mem_w_ck_c2d_w16[4]= W_shift_c2d_w16[32:39];
  assign mem_w_ck_c2d_w16[5]= W_shift_c2d_w16[40:47];
  assign mem_w_ck_c2d_w16[6]= W_shift_c2d_w16[48:55];
  assign mem_w_ck_c2d_w16[7]= W_shift_c2d_w16[56:63];
  assign mem_w_ck_c2d_w16[8]= W_shift_c2d_w16[64:71];
  
  assign mem_w_ck_c2d_w21[0]= W_shift_c2d_w21[0:7];
  assign mem_w_ck_c2d_w21[1]= W_shift_c2d_w21[8:15];
  assign mem_w_ck_c2d_w21[2]= W_shift_c2d_w21[16:23];
  assign mem_w_ck_c2d_w21[3]= W_shift_c2d_w21[24:31];
  assign mem_w_ck_c2d_w21[4]= W_shift_c2d_w21[32:39];
  assign mem_w_ck_c2d_w21[5]= W_shift_c2d_w21[40:47];
  assign mem_w_ck_c2d_w21[6]= W_shift_c2d_w21[48:55];
  assign mem_w_ck_c2d_w21[7]= W_shift_c2d_w21[56:63];
  assign mem_w_ck_c2d_w21[8]= W_shift_c2d_w21[64:71];
  
  assign mem_w_ck_c2d_w22[0]= W_shift_c2d_w22[0:7];
  assign mem_w_ck_c2d_w22[1]= W_shift_c2d_w22[8:15];
  assign mem_w_ck_c2d_w22[2]= W_shift_c2d_w22[16:23];
  assign mem_w_ck_c2d_w22[3]= W_shift_c2d_w22[24:31];
  assign mem_w_ck_c2d_w22[4]= W_shift_c2d_w22[32:39];
  assign mem_w_ck_c2d_w22[5]= W_shift_c2d_w22[40:47];
  assign mem_w_ck_c2d_w22[6]= W_shift_c2d_w22[48:55];
  assign mem_w_ck_c2d_w22[7]= W_shift_c2d_w22[56:63];
  assign mem_w_ck_c2d_w22[8]= W_shift_c2d_w22[64:71];
  
  assign mem_w_ck_c2d_w23[0]= W_shift_c2d_w23[0:7];
  assign mem_w_ck_c2d_w23[1]= W_shift_c2d_w23[8:15];
  assign mem_w_ck_c2d_w23[2]= W_shift_c2d_w23[16:23];
  assign mem_w_ck_c2d_w23[3]= W_shift_c2d_w23[24:31];
  assign mem_w_ck_c2d_w23[4]= W_shift_c2d_w23[32:39];
  assign mem_w_ck_c2d_w23[5]= W_shift_c2d_w23[40:47];
  assign mem_w_ck_c2d_w23[6]= W_shift_c2d_w23[48:55];
  assign mem_w_ck_c2d_w23[7]= W_shift_c2d_w23[56:63];
  assign mem_w_ck_c2d_w23[8]= W_shift_c2d_w23[64:71];
  
  assign mem_w_ck_c2d_w24[0]= W_shift_c2d_w24[0:7];
  assign mem_w_ck_c2d_w24[1]= W_shift_c2d_w24[8:15];
  assign mem_w_ck_c2d_w24[2]= W_shift_c2d_w24[16:23];
  assign mem_w_ck_c2d_w24[3]= W_shift_c2d_w24[24:31];
  assign mem_w_ck_c2d_w24[4]= W_shift_c2d_w24[32:39];
  assign mem_w_ck_c2d_w24[5]= W_shift_c2d_w24[40:47];
  assign mem_w_ck_c2d_w24[6]= W_shift_c2d_w24[48:55];
  assign mem_w_ck_c2d_w24[7]= W_shift_c2d_w24[56:63];
  assign mem_w_ck_c2d_w24[8]= W_shift_c2d_w24[64:71];
  
  assign mem_w_ck_c2d_w25[0]= W_shift_c2d_w25[0:7];
  assign mem_w_ck_c2d_w25[1]= W_shift_c2d_w25[8:15];
  assign mem_w_ck_c2d_w25[2]= W_shift_c2d_w25[16:23];
  assign mem_w_ck_c2d_w25[3]= W_shift_c2d_w25[24:31];
  assign mem_w_ck_c2d_w25[4]= W_shift_c2d_w25[32:39];
  assign mem_w_ck_c2d_w25[5]= W_shift_c2d_w25[40:47];
  assign mem_w_ck_c2d_w25[6]= W_shift_c2d_w25[48:55];
  assign mem_w_ck_c2d_w25[7]= W_shift_c2d_w25[56:63];
  assign mem_w_ck_c2d_w25[8]= W_shift_c2d_w25[64:71];
  
  assign mem_w_ck_c2d_w26[0]= W_shift_c2d_w26[0:7];
  assign mem_w_ck_c2d_w26[1]= W_shift_c2d_w26[8:15];
  assign mem_w_ck_c2d_w26[2]= W_shift_c2d_w26[16:23];
  assign mem_w_ck_c2d_w26[3]= W_shift_c2d_w26[24:31];
  assign mem_w_ck_c2d_w26[4]= W_shift_c2d_w26[32:39];
  assign mem_w_ck_c2d_w26[5]= W_shift_c2d_w26[40:47];
  assign mem_w_ck_c2d_w26[6]= W_shift_c2d_w26[48:55];
  assign mem_w_ck_c2d_w26[7]= W_shift_c2d_w26[56:63];
  assign mem_w_ck_c2d_w26[8]= W_shift_c2d_w26[64:71];
  
  assign mem_w_ck_c2d_w31[0]= W_shift_c2d_w31[0:7];
  assign mem_w_ck_c2d_w31[1]= W_shift_c2d_w31[8:15];
  assign mem_w_ck_c2d_w31[2]= W_shift_c2d_w31[16:23];
  assign mem_w_ck_c2d_w31[3]= W_shift_c2d_w31[24:31];
  assign mem_w_ck_c2d_w31[4]= W_shift_c2d_w31[32:39];
  assign mem_w_ck_c2d_w31[5]= W_shift_c2d_w31[40:47];
  assign mem_w_ck_c2d_w31[6]= W_shift_c2d_w31[48:55];
  assign mem_w_ck_c2d_w31[7]= W_shift_c2d_w31[56:63];
  assign mem_w_ck_c2d_w31[8]= W_shift_c2d_w31[64:71];
  
  assign mem_w_ck_c2d_w32[0]= W_shift_c2d_w32[0:7];
  assign mem_w_ck_c2d_w32[1]= W_shift_c2d_w32[8:15];
  assign mem_w_ck_c2d_w32[2]= W_shift_c2d_w32[16:23];
  assign mem_w_ck_c2d_w32[3]= W_shift_c2d_w32[24:31];
  assign mem_w_ck_c2d_w32[4]= W_shift_c2d_w32[32:39];
  assign mem_w_ck_c2d_w32[5]= W_shift_c2d_w32[40:47];
  assign mem_w_ck_c2d_w32[6]= W_shift_c2d_w32[48:55];
  assign mem_w_ck_c2d_w32[7]= W_shift_c2d_w32[56:63];
  assign mem_w_ck_c2d_w32[8]= W_shift_c2d_w32[64:71];
  
  assign mem_w_ck_c2d_w33[0]= W_shift_c2d_w33[0:7];
  assign mem_w_ck_c2d_w33[1]= W_shift_c2d_w33[8:15];
  assign mem_w_ck_c2d_w33[2]= W_shift_c2d_w33[16:23];
  assign mem_w_ck_c2d_w33[3]= W_shift_c2d_w33[24:31];
  assign mem_w_ck_c2d_w33[4]= W_shift_c2d_w33[32:39];
  assign mem_w_ck_c2d_w33[5]= W_shift_c2d_w33[40:47];
  assign mem_w_ck_c2d_w33[6]= W_shift_c2d_w33[48:55];
  assign mem_w_ck_c2d_w33[7]= W_shift_c2d_w33[56:63];
  assign mem_w_ck_c2d_w33[8]= W_shift_c2d_w33[64:71];
  
  assign mem_w_ck_c2d_w34[0]= W_shift_c2d_w34[0:7];
  assign mem_w_ck_c2d_w34[1]= W_shift_c2d_w34[8:15];
  assign mem_w_ck_c2d_w34[2]= W_shift_c2d_w34[16:23];
  assign mem_w_ck_c2d_w34[3]= W_shift_c2d_w34[24:31];
  assign mem_w_ck_c2d_w34[4]= W_shift_c2d_w34[32:39];
  assign mem_w_ck_c2d_w34[5]= W_shift_c2d_w34[40:47];
  assign mem_w_ck_c2d_w34[6]= W_shift_c2d_w34[48:55];
  assign mem_w_ck_c2d_w34[7]= W_shift_c2d_w34[56:63];
  assign mem_w_ck_c2d_w34[8]= W_shift_c2d_w34[64:71];
  
  assign mem_w_ck_c2d_w35[0]= W_shift_c2d_w35[0:7];
  assign mem_w_ck_c2d_w35[1]= W_shift_c2d_w35[8:15];
  assign mem_w_ck_c2d_w35[2]= W_shift_c2d_w35[16:23];
  assign mem_w_ck_c2d_w35[3]= W_shift_c2d_w35[24:31];
  assign mem_w_ck_c2d_w35[4]= W_shift_c2d_w35[32:39];
  assign mem_w_ck_c2d_w35[5]= W_shift_c2d_w35[40:47];
  assign mem_w_ck_c2d_w35[6]= W_shift_c2d_w35[48:55];
  assign mem_w_ck_c2d_w35[7]= W_shift_c2d_w35[56:63];
  assign mem_w_ck_c2d_w35[8]= W_shift_c2d_w35[64:71];
  
  assign mem_w_ck_c2d_w36[0]= W_shift_c2d_w36[0:7];
  assign mem_w_ck_c2d_w36[1]= W_shift_c2d_w36[8:15];
  assign mem_w_ck_c2d_w36[2]= W_shift_c2d_w36[16:23];
  assign mem_w_ck_c2d_w36[3]= W_shift_c2d_w36[24:31];
  assign mem_w_ck_c2d_w36[4]= W_shift_c2d_w36[32:39];
  assign mem_w_ck_c2d_w36[5]= W_shift_c2d_w36[40:47];
  assign mem_w_ck_c2d_w36[6]= W_shift_c2d_w36[48:55];
  assign mem_w_ck_c2d_w36[7]= W_shift_c2d_w36[56:63];
  assign mem_w_ck_c2d_w36[8]= W_shift_c2d_w36[64:71];
  
  assign mem_w_ck_c2d_w41[0]= W_shift_c2d_w41[0:7];
  assign mem_w_ck_c2d_w41[1]= W_shift_c2d_w41[8:15];
  assign mem_w_ck_c2d_w41[2]= W_shift_c2d_w41[16:23];
  assign mem_w_ck_c2d_w41[3]= W_shift_c2d_w41[24:31];
  assign mem_w_ck_c2d_w41[4]= W_shift_c2d_w41[32:39];
  assign mem_w_ck_c2d_w41[5]= W_shift_c2d_w41[40:47];
  assign mem_w_ck_c2d_w41[6]= W_shift_c2d_w41[48:55];
  assign mem_w_ck_c2d_w41[7]= W_shift_c2d_w41[56:63];
  assign mem_w_ck_c2d_w41[8]= W_shift_c2d_w41[64:71];
  
  assign mem_w_ck_c2d_w42[0]= W_shift_c2d_w42[0:7];
  assign mem_w_ck_c2d_w42[1]= W_shift_c2d_w42[8:15];
  assign mem_w_ck_c2d_w42[2]= W_shift_c2d_w42[16:23];
  assign mem_w_ck_c2d_w42[3]= W_shift_c2d_w42[24:31];
  assign mem_w_ck_c2d_w42[4]= W_shift_c2d_w42[32:39];
  assign mem_w_ck_c2d_w42[5]= W_shift_c2d_w42[40:47];
  assign mem_w_ck_c2d_w42[6]= W_shift_c2d_w42[48:55];
  assign mem_w_ck_c2d_w42[7]= W_shift_c2d_w42[56:63];
  assign mem_w_ck_c2d_w42[8]= W_shift_c2d_w42[64:71];
  
  assign mem_w_ck_c2d_w43[0]= W_shift_c2d_w43[0:7];
  assign mem_w_ck_c2d_w43[1]= W_shift_c2d_w43[8:15];
  assign mem_w_ck_c2d_w43[2]= W_shift_c2d_w43[16:23];
  assign mem_w_ck_c2d_w43[3]= W_shift_c2d_w43[24:31];
  assign mem_w_ck_c2d_w43[4]= W_shift_c2d_w43[32:39];
  assign mem_w_ck_c2d_w43[5]= W_shift_c2d_w43[40:47];
  assign mem_w_ck_c2d_w43[6]= W_shift_c2d_w43[48:55];
  assign mem_w_ck_c2d_w43[7]= W_shift_c2d_w43[56:63];
  assign mem_w_ck_c2d_w43[8]= W_shift_c2d_w43[64:71];
  
  assign mem_w_ck_c2d_w44[0]= W_shift_c2d_w44[0:7];
  assign mem_w_ck_c2d_w44[1]= W_shift_c2d_w44[8:15];
  assign mem_w_ck_c2d_w44[2]= W_shift_c2d_w44[16:23];
  assign mem_w_ck_c2d_w44[3]= W_shift_c2d_w44[24:31];
  assign mem_w_ck_c2d_w44[4]= W_shift_c2d_w44[32:39];
  assign mem_w_ck_c2d_w44[5]= W_shift_c2d_w44[40:47];
  assign mem_w_ck_c2d_w44[6]= W_shift_c2d_w44[48:55];
  assign mem_w_ck_c2d_w44[7]= W_shift_c2d_w44[56:63];
  assign mem_w_ck_c2d_w44[8]= W_shift_c2d_w44[64:71];
  
  assign mem_w_ck_c2d_w45[0]= W_shift_c2d_w45[0:7];
  assign mem_w_ck_c2d_w45[1]= W_shift_c2d_w45[8:15];
  assign mem_w_ck_c2d_w45[2]= W_shift_c2d_w45[16:23];
  assign mem_w_ck_c2d_w45[3]= W_shift_c2d_w45[24:31];
  assign mem_w_ck_c2d_w45[4]= W_shift_c2d_w45[32:39];
  assign mem_w_ck_c2d_w45[5]= W_shift_c2d_w45[40:47];
  assign mem_w_ck_c2d_w45[6]= W_shift_c2d_w45[48:55];
  assign mem_w_ck_c2d_w45[7]= W_shift_c2d_w45[56:63];
  assign mem_w_ck_c2d_w45[8]= W_shift_c2d_w45[64:71];
  
  assign mem_w_ck_c2d_w46[0]= W_shift_c2d_w46[0:7];
  assign mem_w_ck_c2d_w46[1]= W_shift_c2d_w46[8:15];
  assign mem_w_ck_c2d_w46[2]= W_shift_c2d_w46[16:23];
  assign mem_w_ck_c2d_w46[3]= W_shift_c2d_w46[24:31];
  assign mem_w_ck_c2d_w46[4]= W_shift_c2d_w46[32:39];
  assign mem_w_ck_c2d_w46[5]= W_shift_c2d_w46[40:47];
  assign mem_w_ck_c2d_w46[6]= W_shift_c2d_w46[48:55];
  assign mem_w_ck_c2d_w46[7]= W_shift_c2d_w46[56:63];
  assign mem_w_ck_c2d_w46[8]= W_shift_c2d_w46[64:71];
  
  assign mem_w_ck_c2d_w51[0]= W_shift_c2d_w51[0:7];
  assign mem_w_ck_c2d_w51[1]= W_shift_c2d_w51[8:15];
  assign mem_w_ck_c2d_w51[2]= W_shift_c2d_w51[16:23];
  assign mem_w_ck_c2d_w51[3]= W_shift_c2d_w51[24:31];
  assign mem_w_ck_c2d_w51[4]= W_shift_c2d_w51[32:39];
  assign mem_w_ck_c2d_w51[5]= W_shift_c2d_w51[40:47];
  assign mem_w_ck_c2d_w51[6]= W_shift_c2d_w51[48:55];
  assign mem_w_ck_c2d_w51[7]= W_shift_c2d_w51[56:63];
  assign mem_w_ck_c2d_w51[8]= W_shift_c2d_w51[64:71];
  
  assign mem_w_ck_c2d_w52[0]= W_shift_c2d_w52[0:7];
  assign mem_w_ck_c2d_w52[1]= W_shift_c2d_w52[8:15];
  assign mem_w_ck_c2d_w52[2]= W_shift_c2d_w52[16:23];
  assign mem_w_ck_c2d_w52[3]= W_shift_c2d_w52[24:31];
  assign mem_w_ck_c2d_w52[4]= W_shift_c2d_w52[32:39];
  assign mem_w_ck_c2d_w52[5]= W_shift_c2d_w52[40:47];
  assign mem_w_ck_c2d_w52[6]= W_shift_c2d_w52[48:55];
  assign mem_w_ck_c2d_w52[7]= W_shift_c2d_w52[56:63];
  assign mem_w_ck_c2d_w52[8]= W_shift_c2d_w52[64:71];
  
  assign mem_w_ck_c2d_w53[0]= W_shift_c2d_w53[0:7];
  assign mem_w_ck_c2d_w53[1]= W_shift_c2d_w53[8:15];
  assign mem_w_ck_c2d_w53[2]= W_shift_c2d_w53[16:23];
  assign mem_w_ck_c2d_w53[3]= W_shift_c2d_w53[24:31];
  assign mem_w_ck_c2d_w53[4]= W_shift_c2d_w53[32:39];
  assign mem_w_ck_c2d_w53[5]= W_shift_c2d_w53[40:47];
  assign mem_w_ck_c2d_w53[6]= W_shift_c2d_w53[48:55];
  assign mem_w_ck_c2d_w53[7]= W_shift_c2d_w53[56:63];
  assign mem_w_ck_c2d_w53[8]= W_shift_c2d_w53[64:71];
  
  assign mem_w_ck_c2d_w54[0]= W_shift_c2d_w54[0:7];
  assign mem_w_ck_c2d_w54[1]= W_shift_c2d_w54[8:15];
  assign mem_w_ck_c2d_w54[2]= W_shift_c2d_w54[16:23];
  assign mem_w_ck_c2d_w54[3]= W_shift_c2d_w54[24:31];
  assign mem_w_ck_c2d_w54[4]= W_shift_c2d_w54[32:39];
  assign mem_w_ck_c2d_w54[5]= W_shift_c2d_w54[40:47];
  assign mem_w_ck_c2d_w54[6]= W_shift_c2d_w54[48:55];
  assign mem_w_ck_c2d_w54[7]= W_shift_c2d_w54[56:63];
  assign mem_w_ck_c2d_w54[8]= W_shift_c2d_w54[64:71];
  
  assign mem_w_ck_c2d_w55[0]= W_shift_c2d_w55[0:7];
  assign mem_w_ck_c2d_w55[1]= W_shift_c2d_w55[8:15];
  assign mem_w_ck_c2d_w55[2]= W_shift_c2d_w55[16:23];
  assign mem_w_ck_c2d_w55[3]= W_shift_c2d_w55[24:31];
  assign mem_w_ck_c2d_w55[4]= W_shift_c2d_w55[32:39];
  assign mem_w_ck_c2d_w55[5]= W_shift_c2d_w55[40:47];
  assign mem_w_ck_c2d_w55[6]= W_shift_c2d_w55[48:55];
  assign mem_w_ck_c2d_w55[7]= W_shift_c2d_w55[56:63];
  assign mem_w_ck_c2d_w55[8]= W_shift_c2d_w55[64:71];
  
  assign mem_w_ck_c2d_w56[0]= W_shift_c2d_w56[0:7];
  assign mem_w_ck_c2d_w56[1]= W_shift_c2d_w56[8:15];
  assign mem_w_ck_c2d_w56[2]= W_shift_c2d_w56[16:23];
  assign mem_w_ck_c2d_w56[3]= W_shift_c2d_w56[24:31];
  assign mem_w_ck_c2d_w56[4]= W_shift_c2d_w56[32:39];
  assign mem_w_ck_c2d_w56[5]= W_shift_c2d_w56[40:47];
  assign mem_w_ck_c2d_w56[6]= W_shift_c2d_w56[48:55];
  assign mem_w_ck_c2d_w56[7]= W_shift_c2d_w56[56:63];
  assign mem_w_ck_c2d_w56[8]= W_shift_c2d_w56[64:71];
  
  assign mem_w_ck_c2d_w61[0]= W_shift_c2d_w61[0:7];
  assign mem_w_ck_c2d_w61[1]= W_shift_c2d_w61[8:15];
  assign mem_w_ck_c2d_w61[2]= W_shift_c2d_w61[16:23];
  assign mem_w_ck_c2d_w61[3]= W_shift_c2d_w61[24:31];
  assign mem_w_ck_c2d_w61[4]= W_shift_c2d_w61[32:39];
  assign mem_w_ck_c2d_w61[5]= W_shift_c2d_w61[40:47];
  assign mem_w_ck_c2d_w61[6]= W_shift_c2d_w61[48:55];
  assign mem_w_ck_c2d_w61[7]= W_shift_c2d_w61[56:63];
  assign mem_w_ck_c2d_w61[8]= W_shift_c2d_w61[64:71];
  
  assign mem_w_ck_c2d_w62[0]= W_shift_c2d_w62[0:7];
  assign mem_w_ck_c2d_w62[1]= W_shift_c2d_w62[8:15];
  assign mem_w_ck_c2d_w62[2]= W_shift_c2d_w62[16:23];
  assign mem_w_ck_c2d_w62[3]= W_shift_c2d_w62[24:31];
  assign mem_w_ck_c2d_w62[4]= W_shift_c2d_w62[32:39];
  assign mem_w_ck_c2d_w62[5]= W_shift_c2d_w62[40:47];
  assign mem_w_ck_c2d_w62[6]= W_shift_c2d_w62[48:55];
  assign mem_w_ck_c2d_w62[7]= W_shift_c2d_w62[56:63];
  assign mem_w_ck_c2d_w62[8]= W_shift_c2d_w62[64:71];
  
  assign mem_w_ck_c2d_w63[0]= W_shift_c2d_w63[0:7];
  assign mem_w_ck_c2d_w63[1]= W_shift_c2d_w63[8:15];
  assign mem_w_ck_c2d_w63[2]= W_shift_c2d_w63[16:23];
  assign mem_w_ck_c2d_w63[3]= W_shift_c2d_w63[24:31];
  assign mem_w_ck_c2d_w63[4]= W_shift_c2d_w63[32:39];
  assign mem_w_ck_c2d_w63[5]= W_shift_c2d_w63[40:47];
  assign mem_w_ck_c2d_w63[6]= W_shift_c2d_w63[48:55];
  assign mem_w_ck_c2d_w63[7]= W_shift_c2d_w63[56:63];
  assign mem_w_ck_c2d_w63[8]= W_shift_c2d_w63[64:71];
  
  assign mem_w_ck_c2d_w64[0]= W_shift_c2d_w64[0:7];
  assign mem_w_ck_c2d_w64[1]= W_shift_c2d_w64[8:15];
  assign mem_w_ck_c2d_w64[2]= W_shift_c2d_w64[16:23];
  assign mem_w_ck_c2d_w64[3]= W_shift_c2d_w64[24:31];
  assign mem_w_ck_c2d_w64[4]= W_shift_c2d_w64[32:39];
  assign mem_w_ck_c2d_w64[5]= W_shift_c2d_w64[40:47];
  assign mem_w_ck_c2d_w64[6]= W_shift_c2d_w64[48:55];
  assign mem_w_ck_c2d_w64[7]= W_shift_c2d_w64[56:63];
  assign mem_w_ck_c2d_w64[8]= W_shift_c2d_w64[64:71];
  
  assign mem_w_ck_c2d_w65[0]= W_shift_c2d_w65[0:7];
  assign mem_w_ck_c2d_w65[1]= W_shift_c2d_w65[8:15];
  assign mem_w_ck_c2d_w65[2]= W_shift_c2d_w65[16:23];
  assign mem_w_ck_c2d_w65[3]= W_shift_c2d_w65[24:31];
  assign mem_w_ck_c2d_w65[4]= W_shift_c2d_w65[32:39];
  assign mem_w_ck_c2d_w65[5]= W_shift_c2d_w65[40:47];
  assign mem_w_ck_c2d_w65[6]= W_shift_c2d_w65[48:55];
  assign mem_w_ck_c2d_w65[7]= W_shift_c2d_w65[56:63];
  assign mem_w_ck_c2d_w65[8]= W_shift_c2d_w65[64:71];
  
  assign mem_w_ck_c2d_w66[0]= W_shift_c2d_w66[0:7];
  assign mem_w_ck_c2d_w66[1]= W_shift_c2d_w66[8:15];
  assign mem_w_ck_c2d_w66[2]= W_shift_c2d_w66[16:23];
  assign mem_w_ck_c2d_w66[3]= W_shift_c2d_w66[24:31];
  assign mem_w_ck_c2d_w66[4]= W_shift_c2d_w66[32:39];
  assign mem_w_ck_c2d_w66[5]= W_shift_c2d_w66[40:47];
  assign mem_w_ck_c2d_w66[6]= W_shift_c2d_w66[48:55];
  assign mem_w_ck_c2d_w66[7]= W_shift_c2d_w66[56:63];
  assign mem_w_ck_c2d_w66[8]= W_shift_c2d_w66[64:71];
  
  assign mem_w_ck_c2d_w71[0]= W_shift_c2d_w71[0:7];
  assign mem_w_ck_c2d_w71[1]= W_shift_c2d_w71[8:15];
  assign mem_w_ck_c2d_w71[2]= W_shift_c2d_w71[16:23];
  assign mem_w_ck_c2d_w71[3]= W_shift_c2d_w71[24:31];
  assign mem_w_ck_c2d_w71[4]= W_shift_c2d_w71[32:39];
  assign mem_w_ck_c2d_w71[5]= W_shift_c2d_w71[40:47];
  assign mem_w_ck_c2d_w71[6]= W_shift_c2d_w71[48:55];
  assign mem_w_ck_c2d_w71[7]= W_shift_c2d_w71[56:63];
  assign mem_w_ck_c2d_w71[8]= W_shift_c2d_w71[64:71];
  
  assign mem_w_ck_c2d_w72[0]= W_shift_c2d_w72[0:7];
  assign mem_w_ck_c2d_w72[1]= W_shift_c2d_w72[8:15];
  assign mem_w_ck_c2d_w72[2]= W_shift_c2d_w72[16:23];
  assign mem_w_ck_c2d_w72[3]= W_shift_c2d_w72[24:31];
  assign mem_w_ck_c2d_w72[4]= W_shift_c2d_w72[32:39];
  assign mem_w_ck_c2d_w72[5]= W_shift_c2d_w72[40:47];
  assign mem_w_ck_c2d_w72[6]= W_shift_c2d_w72[48:55];
  assign mem_w_ck_c2d_w72[7]= W_shift_c2d_w72[56:63];
  assign mem_w_ck_c2d_w72[8]= W_shift_c2d_w72[64:71];
  
  assign mem_w_ck_c2d_w73[0]= W_shift_c2d_w73[0:7];
  assign mem_w_ck_c2d_w73[1]= W_shift_c2d_w73[8:15];
  assign mem_w_ck_c2d_w73[2]= W_shift_c2d_w73[16:23];
  assign mem_w_ck_c2d_w73[3]= W_shift_c2d_w73[24:31];
  assign mem_w_ck_c2d_w73[4]= W_shift_c2d_w73[32:39];
  assign mem_w_ck_c2d_w73[5]= W_shift_c2d_w73[40:47];
  assign mem_w_ck_c2d_w73[6]= W_shift_c2d_w73[48:55];
  assign mem_w_ck_c2d_w73[7]= W_shift_c2d_w73[56:63];
  assign mem_w_ck_c2d_w73[8]= W_shift_c2d_w73[64:71];
  
  assign mem_w_ck_c2d_w74[0]= W_shift_c2d_w74[0:7];
  assign mem_w_ck_c2d_w74[1]= W_shift_c2d_w74[8:15];
  assign mem_w_ck_c2d_w74[2]= W_shift_c2d_w74[16:23];
  assign mem_w_ck_c2d_w74[3]= W_shift_c2d_w74[24:31];
  assign mem_w_ck_c2d_w74[4]= W_shift_c2d_w74[32:39];
  assign mem_w_ck_c2d_w74[5]= W_shift_c2d_w74[40:47];
  assign mem_w_ck_c2d_w74[6]= W_shift_c2d_w74[48:55];
  assign mem_w_ck_c2d_w74[7]= W_shift_c2d_w74[56:63];
  assign mem_w_ck_c2d_w74[8]= W_shift_c2d_w74[64:71];
  
  assign mem_w_ck_c2d_w75[0]= W_shift_c2d_w75[0:7];
  assign mem_w_ck_c2d_w75[1]= W_shift_c2d_w75[8:15];
  assign mem_w_ck_c2d_w75[2]= W_shift_c2d_w75[16:23];
  assign mem_w_ck_c2d_w75[3]= W_shift_c2d_w75[24:31];
  assign mem_w_ck_c2d_w75[4]= W_shift_c2d_w75[32:39];
  assign mem_w_ck_c2d_w75[5]= W_shift_c2d_w75[40:47];
  assign mem_w_ck_c2d_w75[6]= W_shift_c2d_w75[48:55];
  assign mem_w_ck_c2d_w75[7]= W_shift_c2d_w75[56:63];
  assign mem_w_ck_c2d_w75[8]= W_shift_c2d_w75[64:71];
  
  assign mem_w_ck_c2d_w76[0]= W_shift_c2d_w76[0:7];
  assign mem_w_ck_c2d_w76[1]= W_shift_c2d_w76[8:15];
  assign mem_w_ck_c2d_w76[2]= W_shift_c2d_w76[16:23];
  assign mem_w_ck_c2d_w76[3]= W_shift_c2d_w76[24:31];
  assign mem_w_ck_c2d_w76[4]= W_shift_c2d_w76[32:39];
  assign mem_w_ck_c2d_w76[5]= W_shift_c2d_w76[40:47];
  assign mem_w_ck_c2d_w76[6]= W_shift_c2d_w76[48:55];
  assign mem_w_ck_c2d_w76[7]= W_shift_c2d_w76[56:63];
  assign mem_w_ck_c2d_w76[8]= W_shift_c2d_w76[64:71];
  
  assign mem_w_ck_c2d_w81[0]= W_shift_c2d_w81[0:7];
  assign mem_w_ck_c2d_w81[1]= W_shift_c2d_w81[8:15];
  assign mem_w_ck_c2d_w81[2]= W_shift_c2d_w81[16:23];
  assign mem_w_ck_c2d_w81[3]= W_shift_c2d_w81[24:31];
  assign mem_w_ck_c2d_w81[4]= W_shift_c2d_w81[32:39];
  assign mem_w_ck_c2d_w81[5]= W_shift_c2d_w81[40:47];
  assign mem_w_ck_c2d_w81[6]= W_shift_c2d_w81[48:55];
  assign mem_w_ck_c2d_w81[7]= W_shift_c2d_w81[56:63];
  assign mem_w_ck_c2d_w81[8]= W_shift_c2d_w81[64:71];
  
  assign mem_w_ck_c2d_w82[0]= W_shift_c2d_w82[0:7];
  assign mem_w_ck_c2d_w82[1]= W_shift_c2d_w82[8:15];
  assign mem_w_ck_c2d_w82[2]= W_shift_c2d_w82[16:23];
  assign mem_w_ck_c2d_w82[3]= W_shift_c2d_w82[24:31];
  assign mem_w_ck_c2d_w82[4]= W_shift_c2d_w82[32:39];
  assign mem_w_ck_c2d_w82[5]= W_shift_c2d_w82[40:47];
  assign mem_w_ck_c2d_w82[6]= W_shift_c2d_w82[48:55];
  assign mem_w_ck_c2d_w82[7]= W_shift_c2d_w82[56:63];
  assign mem_w_ck_c2d_w82[8]= W_shift_c2d_w82[64:71];
  
  assign mem_w_ck_c2d_w83[0]= W_shift_c2d_w83[0:7];
  assign mem_w_ck_c2d_w83[1]= W_shift_c2d_w83[8:15];
  assign mem_w_ck_c2d_w83[2]= W_shift_c2d_w83[16:23];
  assign mem_w_ck_c2d_w83[3]= W_shift_c2d_w83[24:31];
  assign mem_w_ck_c2d_w83[4]= W_shift_c2d_w83[32:39];
  assign mem_w_ck_c2d_w83[5]= W_shift_c2d_w83[40:47];
  assign mem_w_ck_c2d_w83[6]= W_shift_c2d_w83[48:55];
  assign mem_w_ck_c2d_w83[7]= W_shift_c2d_w83[56:63];
  assign mem_w_ck_c2d_w83[8]= W_shift_c2d_w83[64:71];
  
  assign mem_w_ck_c2d_w84[0]= W_shift_c2d_w84[0:7];
  assign mem_w_ck_c2d_w84[1]= W_shift_c2d_w84[8:15];
  assign mem_w_ck_c2d_w84[2]= W_shift_c2d_w84[16:23];
  assign mem_w_ck_c2d_w84[3]= W_shift_c2d_w84[24:31];
  assign mem_w_ck_c2d_w84[4]= W_shift_c2d_w84[32:39];
  assign mem_w_ck_c2d_w84[5]= W_shift_c2d_w84[40:47];
  assign mem_w_ck_c2d_w84[6]= W_shift_c2d_w84[48:55];
  assign mem_w_ck_c2d_w84[7]= W_shift_c2d_w84[56:63];
  assign mem_w_ck_c2d_w84[8]= W_shift_c2d_w84[64:71];
  
  assign mem_w_ck_c2d_w85[0]= W_shift_c2d_w85[0:7];
  assign mem_w_ck_c2d_w85[1]= W_shift_c2d_w85[8:15];
  assign mem_w_ck_c2d_w85[2]= W_shift_c2d_w85[16:23];
  assign mem_w_ck_c2d_w85[3]= W_shift_c2d_w85[24:31];
  assign mem_w_ck_c2d_w85[4]= W_shift_c2d_w85[32:39];
  assign mem_w_ck_c2d_w85[5]= W_shift_c2d_w85[40:47];
  assign mem_w_ck_c2d_w85[6]= W_shift_c2d_w85[48:55];
  assign mem_w_ck_c2d_w85[7]= W_shift_c2d_w85[56:63];
  assign mem_w_ck_c2d_w85[8]= W_shift_c2d_w85[64:71];
  
  assign mem_w_ck_c2d_w86[0]= W_shift_c2d_w86[0:7];
  assign mem_w_ck_c2d_w86[1]= W_shift_c2d_w86[8:15];
  assign mem_w_ck_c2d_w86[2]= W_shift_c2d_w86[16:23];
  assign mem_w_ck_c2d_w86[3]= W_shift_c2d_w86[24:31];
  assign mem_w_ck_c2d_w86[4]= W_shift_c2d_w86[32:39];
  assign mem_w_ck_c2d_w86[5]= W_shift_c2d_w86[40:47];
  assign mem_w_ck_c2d_w86[6]= W_shift_c2d_w86[48:55];
  assign mem_w_ck_c2d_w86[7]= W_shift_c2d_w86[56:63];
  assign mem_w_ck_c2d_w86[8]= W_shift_c2d_w86[64:71];
  
  assign mem_w_ck_c2d_w91[0]= W_shift_c2d_w91[0:7];
  assign mem_w_ck_c2d_w91[1]= W_shift_c2d_w91[8:15];
  assign mem_w_ck_c2d_w91[2]= W_shift_c2d_w91[16:23];
  assign mem_w_ck_c2d_w91[3]= W_shift_c2d_w91[24:31];
  assign mem_w_ck_c2d_w91[4]= W_shift_c2d_w91[32:39];
  assign mem_w_ck_c2d_w91[5]= W_shift_c2d_w91[40:47];
  assign mem_w_ck_c2d_w91[6]= W_shift_c2d_w91[48:55];
  assign mem_w_ck_c2d_w91[7]= W_shift_c2d_w91[56:63];
  assign mem_w_ck_c2d_w91[8]= W_shift_c2d_w91[64:71];
  
  assign mem_w_ck_c2d_w92[0]= W_shift_c2d_w92[0:7];
  assign mem_w_ck_c2d_w92[1]= W_shift_c2d_w92[8:15];
  assign mem_w_ck_c2d_w92[2]= W_shift_c2d_w92[16:23];
  assign mem_w_ck_c2d_w92[3]= W_shift_c2d_w92[24:31];
  assign mem_w_ck_c2d_w92[4]= W_shift_c2d_w92[32:39];
  assign mem_w_ck_c2d_w92[5]= W_shift_c2d_w92[40:47];
  assign mem_w_ck_c2d_w92[6]= W_shift_c2d_w92[48:55];
  assign mem_w_ck_c2d_w92[7]= W_shift_c2d_w92[56:63];
  assign mem_w_ck_c2d_w92[8]= W_shift_c2d_w92[64:71];
  
  assign mem_w_ck_c2d_w93[0]= W_shift_c2d_w93[0:7];
  assign mem_w_ck_c2d_w93[1]= W_shift_c2d_w93[8:15];
  assign mem_w_ck_c2d_w93[2]= W_shift_c2d_w93[16:23];
  assign mem_w_ck_c2d_w93[3]= W_shift_c2d_w93[24:31];
  assign mem_w_ck_c2d_w93[4]= W_shift_c2d_w93[32:39];
  assign mem_w_ck_c2d_w93[5]= W_shift_c2d_w93[40:47];
  assign mem_w_ck_c2d_w93[6]= W_shift_c2d_w93[48:55];
  assign mem_w_ck_c2d_w93[7]= W_shift_c2d_w93[56:63];
  assign mem_w_ck_c2d_w93[8]= W_shift_c2d_w93[64:71];
  
  assign mem_w_ck_c2d_w94[0]= W_shift_c2d_w94[0:7];
  assign mem_w_ck_c2d_w94[1]= W_shift_c2d_w94[8:15];
  assign mem_w_ck_c2d_w94[2]= W_shift_c2d_w94[16:23];
  assign mem_w_ck_c2d_w94[3]= W_shift_c2d_w94[24:31];
  assign mem_w_ck_c2d_w94[4]= W_shift_c2d_w94[32:39];
  assign mem_w_ck_c2d_w94[5]= W_shift_c2d_w94[40:47];
  assign mem_w_ck_c2d_w94[6]= W_shift_c2d_w94[48:55];
  assign mem_w_ck_c2d_w94[7]= W_shift_c2d_w94[56:63];
  assign mem_w_ck_c2d_w94[8]= W_shift_c2d_w94[64:71];
  
  assign mem_w_ck_c2d_w95[0]= W_shift_c2d_w95[0:7];
  assign mem_w_ck_c2d_w95[1]= W_shift_c2d_w95[8:15];
  assign mem_w_ck_c2d_w95[2]= W_shift_c2d_w95[16:23];
  assign mem_w_ck_c2d_w95[3]= W_shift_c2d_w95[24:31];
  assign mem_w_ck_c2d_w95[4]= W_shift_c2d_w95[32:39];
  assign mem_w_ck_c2d_w95[5]= W_shift_c2d_w95[40:47];
  assign mem_w_ck_c2d_w95[6]= W_shift_c2d_w95[48:55];
  assign mem_w_ck_c2d_w95[7]= W_shift_c2d_w95[56:63];
  assign mem_w_ck_c2d_w95[8]= W_shift_c2d_w95[64:71];
  
  assign mem_w_ck_c2d_w96[0]= W_shift_c2d_w96[0:7];
  assign mem_w_ck_c2d_w96[1]= W_shift_c2d_w96[8:15];
  assign mem_w_ck_c2d_w96[2]= W_shift_c2d_w96[16:23];
  assign mem_w_ck_c2d_w96[3]= W_shift_c2d_w96[24:31];
  assign mem_w_ck_c2d_w96[4]= W_shift_c2d_w96[32:39];
  assign mem_w_ck_c2d_w96[5]= W_shift_c2d_w96[40:47];
  assign mem_w_ck_c2d_w96[6]= W_shift_c2d_w96[48:55];
  assign mem_w_ck_c2d_w96[7]= W_shift_c2d_w96[56:63];
  assign mem_w_ck_c2d_w96[8]= W_shift_c2d_w96[64:71];
  
  assign mem_w_ck_c2d_w101[0]= W_shift_c2d_w101[0:7];
  assign mem_w_ck_c2d_w101[1]= W_shift_c2d_w101[8:15];
  assign mem_w_ck_c2d_w101[2]= W_shift_c2d_w101[16:23];
  assign mem_w_ck_c2d_w101[3]= W_shift_c2d_w101[24:31];
  assign mem_w_ck_c2d_w101[4]= W_shift_c2d_w101[32:39];
  assign mem_w_ck_c2d_w101[5]= W_shift_c2d_w101[40:47];
  assign mem_w_ck_c2d_w101[6]= W_shift_c2d_w101[48:55];
  assign mem_w_ck_c2d_w101[7]= W_shift_c2d_w101[56:63];
  assign mem_w_ck_c2d_w101[8]= W_shift_c2d_w101[64:71];
  
  assign mem_w_ck_c2d_w102[0]= W_shift_c2d_w102[0:7];
  assign mem_w_ck_c2d_w102[1]= W_shift_c2d_w102[8:15];
  assign mem_w_ck_c2d_w102[2]= W_shift_c2d_w102[16:23];
  assign mem_w_ck_c2d_w102[3]= W_shift_c2d_w102[24:31];
  assign mem_w_ck_c2d_w102[4]= W_shift_c2d_w102[32:39];
  assign mem_w_ck_c2d_w102[5]= W_shift_c2d_w102[40:47];
  assign mem_w_ck_c2d_w102[6]= W_shift_c2d_w102[48:55];
  assign mem_w_ck_c2d_w102[7]= W_shift_c2d_w102[56:63];
  assign mem_w_ck_c2d_w102[8]= W_shift_c2d_w102[64:71];
  
  assign mem_w_ck_c2d_w103[0]= W_shift_c2d_w103[0:7];
  assign mem_w_ck_c2d_w103[1]= W_shift_c2d_w103[8:15];
  assign mem_w_ck_c2d_w103[2]= W_shift_c2d_w103[16:23];
  assign mem_w_ck_c2d_w103[3]= W_shift_c2d_w103[24:31];
  assign mem_w_ck_c2d_w103[4]= W_shift_c2d_w103[32:39];
  assign mem_w_ck_c2d_w103[5]= W_shift_c2d_w103[40:47];
  assign mem_w_ck_c2d_w103[6]= W_shift_c2d_w103[48:55];
  assign mem_w_ck_c2d_w103[7]= W_shift_c2d_w103[56:63];
  assign mem_w_ck_c2d_w103[8]= W_shift_c2d_w103[64:71];
  
  assign mem_w_ck_c2d_w104[0]= W_shift_c2d_w104[0:7];
  assign mem_w_ck_c2d_w104[1]= W_shift_c2d_w104[8:15];
  assign mem_w_ck_c2d_w104[2]= W_shift_c2d_w104[16:23];
  assign mem_w_ck_c2d_w104[3]= W_shift_c2d_w104[24:31];
  assign mem_w_ck_c2d_w104[4]= W_shift_c2d_w104[32:39];
  assign mem_w_ck_c2d_w104[5]= W_shift_c2d_w104[40:47];
  assign mem_w_ck_c2d_w104[6]= W_shift_c2d_w104[48:55];
  assign mem_w_ck_c2d_w104[7]= W_shift_c2d_w104[56:63];
  assign mem_w_ck_c2d_w104[8]= W_shift_c2d_w104[64:71];
  
  assign mem_w_ck_c2d_w105[0]= W_shift_c2d_w105[0:7];
  assign mem_w_ck_c2d_w105[1]= W_shift_c2d_w105[8:15];
  assign mem_w_ck_c2d_w105[2]= W_shift_c2d_w105[16:23];
  assign mem_w_ck_c2d_w105[3]= W_shift_c2d_w105[24:31];
  assign mem_w_ck_c2d_w105[4]= W_shift_c2d_w105[32:39];
  assign mem_w_ck_c2d_w105[5]= W_shift_c2d_w105[40:47];
  assign mem_w_ck_c2d_w105[6]= W_shift_c2d_w105[48:55];
  assign mem_w_ck_c2d_w105[7]= W_shift_c2d_w105[56:63];
  assign mem_w_ck_c2d_w105[8]= W_shift_c2d_w105[64:71];
  
  assign mem_w_ck_c2d_w106[0]= W_shift_c2d_w106[0:7];
  assign mem_w_ck_c2d_w106[1]= W_shift_c2d_w106[8:15];
  assign mem_w_ck_c2d_w106[2]= W_shift_c2d_w106[16:23];
  assign mem_w_ck_c2d_w106[3]= W_shift_c2d_w106[24:31];
  assign mem_w_ck_c2d_w106[4]= W_shift_c2d_w106[32:39];
  assign mem_w_ck_c2d_w106[5]= W_shift_c2d_w106[40:47];
  assign mem_w_ck_c2d_w106[6]= W_shift_c2d_w106[48:55];
  assign mem_w_ck_c2d_w106[7]= W_shift_c2d_w106[56:63];
  assign mem_w_ck_c2d_w106[8]= W_shift_c2d_w106[64:71];
  
  assign mem_w_ck_c2d_w111[0]= W_shift_c2d_w111[0:7];
  assign mem_w_ck_c2d_w111[1]= W_shift_c2d_w111[8:15];
  assign mem_w_ck_c2d_w111[2]= W_shift_c2d_w111[16:23];
  assign mem_w_ck_c2d_w111[3]= W_shift_c2d_w111[24:31];
  assign mem_w_ck_c2d_w111[4]= W_shift_c2d_w111[32:39];
  assign mem_w_ck_c2d_w111[5]= W_shift_c2d_w111[40:47];
  assign mem_w_ck_c2d_w111[6]= W_shift_c2d_w111[48:55];
  assign mem_w_ck_c2d_w111[7]= W_shift_c2d_w111[56:63];
  assign mem_w_ck_c2d_w111[8]= W_shift_c2d_w111[64:71];
  
  assign mem_w_ck_c2d_w112[0]= W_shift_c2d_w112[0:7];
  assign mem_w_ck_c2d_w112[1]= W_shift_c2d_w112[8:15];
  assign mem_w_ck_c2d_w112[2]= W_shift_c2d_w112[16:23];
  assign mem_w_ck_c2d_w112[3]= W_shift_c2d_w112[24:31];
  assign mem_w_ck_c2d_w112[4]= W_shift_c2d_w112[32:39];
  assign mem_w_ck_c2d_w112[5]= W_shift_c2d_w112[40:47];
  assign mem_w_ck_c2d_w112[6]= W_shift_c2d_w112[48:55];
  assign mem_w_ck_c2d_w112[7]= W_shift_c2d_w112[56:63];
  assign mem_w_ck_c2d_w112[8]= W_shift_c2d_w112[64:71];
  
  assign mem_w_ck_c2d_w113[0]= W_shift_c2d_w113[0:7];
  assign mem_w_ck_c2d_w113[1]= W_shift_c2d_w113[8:15];
  assign mem_w_ck_c2d_w113[2]= W_shift_c2d_w113[16:23];
  assign mem_w_ck_c2d_w113[3]= W_shift_c2d_w113[24:31];
  assign mem_w_ck_c2d_w113[4]= W_shift_c2d_w113[32:39];
  assign mem_w_ck_c2d_w113[5]= W_shift_c2d_w113[40:47];
  assign mem_w_ck_c2d_w113[6]= W_shift_c2d_w113[48:55];
  assign mem_w_ck_c2d_w113[7]= W_shift_c2d_w113[56:63];
  assign mem_w_ck_c2d_w113[8]= W_shift_c2d_w113[64:71];
  
  assign mem_w_ck_c2d_w114[0]= W_shift_c2d_w114[0:7];
  assign mem_w_ck_c2d_w114[1]= W_shift_c2d_w114[8:15];
  assign mem_w_ck_c2d_w114[2]= W_shift_c2d_w114[16:23];
  assign mem_w_ck_c2d_w114[3]= W_shift_c2d_w114[24:31];
  assign mem_w_ck_c2d_w114[4]= W_shift_c2d_w114[32:39];
  assign mem_w_ck_c2d_w114[5]= W_shift_c2d_w114[40:47];
  assign mem_w_ck_c2d_w114[6]= W_shift_c2d_w114[48:55];
  assign mem_w_ck_c2d_w114[7]= W_shift_c2d_w114[56:63];
  assign mem_w_ck_c2d_w114[8]= W_shift_c2d_w114[64:71];
  
  assign mem_w_ck_c2d_w115[0]= W_shift_c2d_w115[0:7];
  assign mem_w_ck_c2d_w115[1]= W_shift_c2d_w115[8:15];
  assign mem_w_ck_c2d_w115[2]= W_shift_c2d_w115[16:23];
  assign mem_w_ck_c2d_w115[3]= W_shift_c2d_w115[24:31];
  assign mem_w_ck_c2d_w115[4]= W_shift_c2d_w115[32:39];
  assign mem_w_ck_c2d_w115[5]= W_shift_c2d_w115[40:47];
  assign mem_w_ck_c2d_w115[6]= W_shift_c2d_w115[48:55];
  assign mem_w_ck_c2d_w115[7]= W_shift_c2d_w115[56:63];
  assign mem_w_ck_c2d_w115[8]= W_shift_c2d_w115[64:71];
  
  assign mem_w_ck_c2d_w116[0]= W_shift_c2d_w116[0:7];
  assign mem_w_ck_c2d_w116[1]= W_shift_c2d_w116[8:15];
  assign mem_w_ck_c2d_w116[2]= W_shift_c2d_w116[16:23];
  assign mem_w_ck_c2d_w116[3]= W_shift_c2d_w116[24:31];
  assign mem_w_ck_c2d_w116[4]= W_shift_c2d_w116[32:39];
  assign mem_w_ck_c2d_w116[5]= W_shift_c2d_w116[40:47];
  assign mem_w_ck_c2d_w116[6]= W_shift_c2d_w116[48:55];
  assign mem_w_ck_c2d_w116[7]= W_shift_c2d_w116[56:63];
  assign mem_w_ck_c2d_w116[8]= W_shift_c2d_w116[64:71];
  
  assign mem_w_ck_c2d_w121[0]= W_shift_c2d_w121[0:7];
  assign mem_w_ck_c2d_w121[1]= W_shift_c2d_w121[8:15];
  assign mem_w_ck_c2d_w121[2]= W_shift_c2d_w121[16:23];
  assign mem_w_ck_c2d_w121[3]= W_shift_c2d_w121[24:31];
  assign mem_w_ck_c2d_w121[4]= W_shift_c2d_w121[32:39];
  assign mem_w_ck_c2d_w121[5]= W_shift_c2d_w121[40:47];
  assign mem_w_ck_c2d_w121[6]= W_shift_c2d_w121[48:55];
  assign mem_w_ck_c2d_w121[7]= W_shift_c2d_w121[56:63];
  assign mem_w_ck_c2d_w121[8]= W_shift_c2d_w121[64:71];
  
  assign mem_w_ck_c2d_w122[0]= W_shift_c2d_w122[0:7];
  assign mem_w_ck_c2d_w122[1]= W_shift_c2d_w122[8:15];
  assign mem_w_ck_c2d_w122[2]= W_shift_c2d_w122[16:23];
  assign mem_w_ck_c2d_w122[3]= W_shift_c2d_w122[24:31];
  assign mem_w_ck_c2d_w122[4]= W_shift_c2d_w122[32:39];
  assign mem_w_ck_c2d_w122[5]= W_shift_c2d_w122[40:47];
  assign mem_w_ck_c2d_w122[6]= W_shift_c2d_w122[48:55];
  assign mem_w_ck_c2d_w122[7]= W_shift_c2d_w122[56:63];
  assign mem_w_ck_c2d_w122[8]= W_shift_c2d_w122[64:71];
  
  assign mem_w_ck_c2d_w123[0]= W_shift_c2d_w123[0:7];
  assign mem_w_ck_c2d_w123[1]= W_shift_c2d_w123[8:15];
  assign mem_w_ck_c2d_w123[2]= W_shift_c2d_w123[16:23];
  assign mem_w_ck_c2d_w123[3]= W_shift_c2d_w123[24:31];
  assign mem_w_ck_c2d_w123[4]= W_shift_c2d_w123[32:39];
  assign mem_w_ck_c2d_w123[5]= W_shift_c2d_w123[40:47];
  assign mem_w_ck_c2d_w123[6]= W_shift_c2d_w123[48:55];
  assign mem_w_ck_c2d_w123[7]= W_shift_c2d_w123[56:63];
  assign mem_w_ck_c2d_w123[8]= W_shift_c2d_w123[64:71];
  
  assign mem_w_ck_c2d_w124[0]= W_shift_c2d_w124[0:7];
  assign mem_w_ck_c2d_w124[1]= W_shift_c2d_w124[8:15];
  assign mem_w_ck_c2d_w124[2]= W_shift_c2d_w124[16:23];
  assign mem_w_ck_c2d_w124[3]= W_shift_c2d_w124[24:31];
  assign mem_w_ck_c2d_w124[4]= W_shift_c2d_w124[32:39];
  assign mem_w_ck_c2d_w124[5]= W_shift_c2d_w124[40:47];
  assign mem_w_ck_c2d_w124[6]= W_shift_c2d_w124[48:55];
  assign mem_w_ck_c2d_w124[7]= W_shift_c2d_w124[56:63];
  assign mem_w_ck_c2d_w124[8]= W_shift_c2d_w124[64:71];
  
  assign mem_w_ck_c2d_w125[0]= W_shift_c2d_w125[0:7];
  assign mem_w_ck_c2d_w125[1]= W_shift_c2d_w125[8:15];
  assign mem_w_ck_c2d_w125[2]= W_shift_c2d_w125[16:23];
  assign mem_w_ck_c2d_w125[3]= W_shift_c2d_w125[24:31];
  assign mem_w_ck_c2d_w125[4]= W_shift_c2d_w125[32:39];
  assign mem_w_ck_c2d_w125[5]= W_shift_c2d_w125[40:47];
  assign mem_w_ck_c2d_w125[6]= W_shift_c2d_w125[48:55];
  assign mem_w_ck_c2d_w125[7]= W_shift_c2d_w125[56:63];
  assign mem_w_ck_c2d_w125[8]= W_shift_c2d_w125[64:71];
  
  assign mem_w_ck_c2d_w126[0]= W_shift_c2d_w126[0:7];
  assign mem_w_ck_c2d_w126[1]= W_shift_c2d_w126[8:15];
  assign mem_w_ck_c2d_w126[2]= W_shift_c2d_w126[16:23];
  assign mem_w_ck_c2d_w126[3]= W_shift_c2d_w126[24:31];
  assign mem_w_ck_c2d_w126[4]= W_shift_c2d_w126[32:39];
  assign mem_w_ck_c2d_w126[5]= W_shift_c2d_w126[40:47];
  assign mem_w_ck_c2d_w126[6]= W_shift_c2d_w126[48:55];
  assign mem_w_ck_c2d_w126[7]= W_shift_c2d_w126[56:63];
  assign mem_w_ck_c2d_w126[8]= W_shift_c2d_w126[64:71];
  
  assign mem_w_ck_c2d_w131[0]= W_shift_c2d_w131[0:7];
  assign mem_w_ck_c2d_w131[1]= W_shift_c2d_w131[8:15];
  assign mem_w_ck_c2d_w131[2]= W_shift_c2d_w131[16:23];
  assign mem_w_ck_c2d_w131[3]= W_shift_c2d_w131[24:31];
  assign mem_w_ck_c2d_w131[4]= W_shift_c2d_w131[32:39];
  assign mem_w_ck_c2d_w131[5]= W_shift_c2d_w131[40:47];
  assign mem_w_ck_c2d_w131[6]= W_shift_c2d_w131[48:55];
  assign mem_w_ck_c2d_w131[7]= W_shift_c2d_w131[56:63];
  assign mem_w_ck_c2d_w131[8]= W_shift_c2d_w131[64:71];
  
  assign mem_w_ck_c2d_w132[0]= W_shift_c2d_w132[0:7];
  assign mem_w_ck_c2d_w132[1]= W_shift_c2d_w132[8:15];
  assign mem_w_ck_c2d_w132[2]= W_shift_c2d_w132[16:23];
  assign mem_w_ck_c2d_w132[3]= W_shift_c2d_w132[24:31];
  assign mem_w_ck_c2d_w132[4]= W_shift_c2d_w132[32:39];
  assign mem_w_ck_c2d_w132[5]= W_shift_c2d_w132[40:47];
  assign mem_w_ck_c2d_w132[6]= W_shift_c2d_w132[48:55];
  assign mem_w_ck_c2d_w132[7]= W_shift_c2d_w132[56:63];
  assign mem_w_ck_c2d_w132[8]= W_shift_c2d_w132[64:71];
  
  assign mem_w_ck_c2d_w133[0]= W_shift_c2d_w133[0:7];
  assign mem_w_ck_c2d_w133[1]= W_shift_c2d_w133[8:15];
  assign mem_w_ck_c2d_w133[2]= W_shift_c2d_w133[16:23];
  assign mem_w_ck_c2d_w133[3]= W_shift_c2d_w133[24:31];
  assign mem_w_ck_c2d_w133[4]= W_shift_c2d_w133[32:39];
  assign mem_w_ck_c2d_w133[5]= W_shift_c2d_w133[40:47];
  assign mem_w_ck_c2d_w133[6]= W_shift_c2d_w133[48:55];
  assign mem_w_ck_c2d_w133[7]= W_shift_c2d_w133[56:63];
  assign mem_w_ck_c2d_w133[8]= W_shift_c2d_w133[64:71];
  
  assign mem_w_ck_c2d_w134[0]= W_shift_c2d_w134[0:7];
  assign mem_w_ck_c2d_w134[1]= W_shift_c2d_w134[8:15];
  assign mem_w_ck_c2d_w134[2]= W_shift_c2d_w134[16:23];
  assign mem_w_ck_c2d_w134[3]= W_shift_c2d_w134[24:31];
  assign mem_w_ck_c2d_w134[4]= W_shift_c2d_w134[32:39];
  assign mem_w_ck_c2d_w134[5]= W_shift_c2d_w134[40:47];
  assign mem_w_ck_c2d_w134[6]= W_shift_c2d_w134[48:55];
  assign mem_w_ck_c2d_w134[7]= W_shift_c2d_w134[56:63];
  assign mem_w_ck_c2d_w134[8]= W_shift_c2d_w134[64:71];
  
  assign mem_w_ck_c2d_w135[0]= W_shift_c2d_w135[0:7];
  assign mem_w_ck_c2d_w135[1]= W_shift_c2d_w135[8:15];
  assign mem_w_ck_c2d_w135[2]= W_shift_c2d_w135[16:23];
  assign mem_w_ck_c2d_w135[3]= W_shift_c2d_w135[24:31];
  assign mem_w_ck_c2d_w135[4]= W_shift_c2d_w135[32:39];
  assign mem_w_ck_c2d_w135[5]= W_shift_c2d_w135[40:47];
  assign mem_w_ck_c2d_w135[6]= W_shift_c2d_w135[48:55];
  assign mem_w_ck_c2d_w135[7]= W_shift_c2d_w135[56:63];
  assign mem_w_ck_c2d_w135[8]= W_shift_c2d_w135[64:71];
  
  assign mem_w_ck_c2d_w136[0]= W_shift_c2d_w136[0:7];
  assign mem_w_ck_c2d_w136[1]= W_shift_c2d_w136[8:15];
  assign mem_w_ck_c2d_w136[2]= W_shift_c2d_w136[16:23];
  assign mem_w_ck_c2d_w136[3]= W_shift_c2d_w136[24:31];
  assign mem_w_ck_c2d_w136[4]= W_shift_c2d_w136[32:39];
  assign mem_w_ck_c2d_w136[5]= W_shift_c2d_w136[40:47];
  assign mem_w_ck_c2d_w136[6]= W_shift_c2d_w136[48:55];
  assign mem_w_ck_c2d_w136[7]= W_shift_c2d_w136[56:63];
  assign mem_w_ck_c2d_w136[8]= W_shift_c2d_w136[64:71];
  
  assign mem_w_ck_c2d_w141[0]= W_shift_c2d_w141[0:7];
  assign mem_w_ck_c2d_w141[1]= W_shift_c2d_w141[8:15];
  assign mem_w_ck_c2d_w141[2]= W_shift_c2d_w141[16:23];
  assign mem_w_ck_c2d_w141[3]= W_shift_c2d_w141[24:31];
  assign mem_w_ck_c2d_w141[4]= W_shift_c2d_w141[32:39];
  assign mem_w_ck_c2d_w141[5]= W_shift_c2d_w141[40:47];
  assign mem_w_ck_c2d_w141[6]= W_shift_c2d_w141[48:55];
  assign mem_w_ck_c2d_w141[7]= W_shift_c2d_w141[56:63];
  assign mem_w_ck_c2d_w141[8]= W_shift_c2d_w141[64:71];
  
  assign mem_w_ck_c2d_w142[0]= W_shift_c2d_w142[0:7];
  assign mem_w_ck_c2d_w142[1]= W_shift_c2d_w142[8:15];
  assign mem_w_ck_c2d_w142[2]= W_shift_c2d_w142[16:23];
  assign mem_w_ck_c2d_w142[3]= W_shift_c2d_w142[24:31];
  assign mem_w_ck_c2d_w142[4]= W_shift_c2d_w142[32:39];
  assign mem_w_ck_c2d_w142[5]= W_shift_c2d_w142[40:47];
  assign mem_w_ck_c2d_w142[6]= W_shift_c2d_w142[48:55];
  assign mem_w_ck_c2d_w142[7]= W_shift_c2d_w142[56:63];
  assign mem_w_ck_c2d_w142[8]= W_shift_c2d_w142[64:71];
  
  assign mem_w_ck_c2d_w143[0]= W_shift_c2d_w143[0:7];
  assign mem_w_ck_c2d_w143[1]= W_shift_c2d_w143[8:15];
  assign mem_w_ck_c2d_w143[2]= W_shift_c2d_w143[16:23];
  assign mem_w_ck_c2d_w143[3]= W_shift_c2d_w143[24:31];
  assign mem_w_ck_c2d_w143[4]= W_shift_c2d_w143[32:39];
  assign mem_w_ck_c2d_w143[5]= W_shift_c2d_w143[40:47];
  assign mem_w_ck_c2d_w143[6]= W_shift_c2d_w143[48:55];
  assign mem_w_ck_c2d_w143[7]= W_shift_c2d_w143[56:63];
  assign mem_w_ck_c2d_w143[8]= W_shift_c2d_w143[64:71];
  
  assign mem_w_ck_c2d_w144[0]= W_shift_c2d_w144[0:7];
  assign mem_w_ck_c2d_w144[1]= W_shift_c2d_w144[8:15];
  assign mem_w_ck_c2d_w144[2]= W_shift_c2d_w144[16:23];
  assign mem_w_ck_c2d_w144[3]= W_shift_c2d_w144[24:31];
  assign mem_w_ck_c2d_w144[4]= W_shift_c2d_w144[32:39];
  assign mem_w_ck_c2d_w144[5]= W_shift_c2d_w144[40:47];
  assign mem_w_ck_c2d_w144[6]= W_shift_c2d_w144[48:55];
  assign mem_w_ck_c2d_w144[7]= W_shift_c2d_w144[56:63];
  assign mem_w_ck_c2d_w144[8]= W_shift_c2d_w144[64:71];
  
  assign mem_w_ck_c2d_w145[0]= W_shift_c2d_w145[0:7];
  assign mem_w_ck_c2d_w145[1]= W_shift_c2d_w145[8:15];
  assign mem_w_ck_c2d_w145[2]= W_shift_c2d_w145[16:23];
  assign mem_w_ck_c2d_w145[3]= W_shift_c2d_w145[24:31];
  assign mem_w_ck_c2d_w145[4]= W_shift_c2d_w145[32:39];
  assign mem_w_ck_c2d_w145[5]= W_shift_c2d_w145[40:47];
  assign mem_w_ck_c2d_w145[6]= W_shift_c2d_w145[48:55];
  assign mem_w_ck_c2d_w145[7]= W_shift_c2d_w145[56:63];
  assign mem_w_ck_c2d_w145[8]= W_shift_c2d_w145[64:71];
  
  assign mem_w_ck_c2d_w146[0]= W_shift_c2d_w146[0:7];
  assign mem_w_ck_c2d_w146[1]= W_shift_c2d_w146[8:15];
  assign mem_w_ck_c2d_w146[2]= W_shift_c2d_w146[16:23];
  assign mem_w_ck_c2d_w146[3]= W_shift_c2d_w146[24:31];
  assign mem_w_ck_c2d_w146[4]= W_shift_c2d_w146[32:39];
  assign mem_w_ck_c2d_w146[5]= W_shift_c2d_w146[40:47];
  assign mem_w_ck_c2d_w146[6]= W_shift_c2d_w146[48:55];
  assign mem_w_ck_c2d_w146[7]= W_shift_c2d_w146[56:63];
  assign mem_w_ck_c2d_w146[8]= W_shift_c2d_w146[64:71];
  
  assign mem_w_ck_c2d_w151[0]= W_shift_c2d_w151[0:7];
  assign mem_w_ck_c2d_w151[1]= W_shift_c2d_w151[8:15];
  assign mem_w_ck_c2d_w151[2]= W_shift_c2d_w151[16:23];
  assign mem_w_ck_c2d_w151[3]= W_shift_c2d_w151[24:31];
  assign mem_w_ck_c2d_w151[4]= W_shift_c2d_w151[32:39];
  assign mem_w_ck_c2d_w151[5]= W_shift_c2d_w151[40:47];
  assign mem_w_ck_c2d_w151[6]= W_shift_c2d_w151[48:55];
  assign mem_w_ck_c2d_w151[7]= W_shift_c2d_w151[56:63];
  assign mem_w_ck_c2d_w151[8]= W_shift_c2d_w151[64:71];
  
  assign mem_w_ck_c2d_w152[0]= W_shift_c2d_w152[0:7];
  assign mem_w_ck_c2d_w152[1]= W_shift_c2d_w152[8:15];
  assign mem_w_ck_c2d_w152[2]= W_shift_c2d_w152[16:23];
  assign mem_w_ck_c2d_w152[3]= W_shift_c2d_w152[24:31];
  assign mem_w_ck_c2d_w152[4]= W_shift_c2d_w152[32:39];
  assign mem_w_ck_c2d_w152[5]= W_shift_c2d_w152[40:47];
  assign mem_w_ck_c2d_w152[6]= W_shift_c2d_w152[48:55];
  assign mem_w_ck_c2d_w152[7]= W_shift_c2d_w152[56:63];
  assign mem_w_ck_c2d_w152[8]= W_shift_c2d_w152[64:71];
  
  assign mem_w_ck_c2d_w153[0]= W_shift_c2d_w153[0:7];
  assign mem_w_ck_c2d_w153[1]= W_shift_c2d_w153[8:15];
  assign mem_w_ck_c2d_w153[2]= W_shift_c2d_w153[16:23];
  assign mem_w_ck_c2d_w153[3]= W_shift_c2d_w153[24:31];
  assign mem_w_ck_c2d_w153[4]= W_shift_c2d_w153[32:39];
  assign mem_w_ck_c2d_w153[5]= W_shift_c2d_w153[40:47];
  assign mem_w_ck_c2d_w153[6]= W_shift_c2d_w153[48:55];
  assign mem_w_ck_c2d_w153[7]= W_shift_c2d_w153[56:63];
  assign mem_w_ck_c2d_w153[8]= W_shift_c2d_w153[64:71];
  
  assign mem_w_ck_c2d_w154[0]= W_shift_c2d_w154[0:7];
  assign mem_w_ck_c2d_w154[1]= W_shift_c2d_w154[8:15];
  assign mem_w_ck_c2d_w154[2]= W_shift_c2d_w154[16:23];
  assign mem_w_ck_c2d_w154[3]= W_shift_c2d_w154[24:31];
  assign mem_w_ck_c2d_w154[4]= W_shift_c2d_w154[32:39];
  assign mem_w_ck_c2d_w154[5]= W_shift_c2d_w154[40:47];
  assign mem_w_ck_c2d_w154[6]= W_shift_c2d_w154[48:55];
  assign mem_w_ck_c2d_w154[7]= W_shift_c2d_w154[56:63];
  assign mem_w_ck_c2d_w154[8]= W_shift_c2d_w154[64:71];
  
  assign mem_w_ck_c2d_w155[0]= W_shift_c2d_w155[0:7];
  assign mem_w_ck_c2d_w155[1]= W_shift_c2d_w155[8:15];
  assign mem_w_ck_c2d_w155[2]= W_shift_c2d_w155[16:23];
  assign mem_w_ck_c2d_w155[3]= W_shift_c2d_w155[24:31];
  assign mem_w_ck_c2d_w155[4]= W_shift_c2d_w155[32:39];
  assign mem_w_ck_c2d_w155[5]= W_shift_c2d_w155[40:47];
  assign mem_w_ck_c2d_w155[6]= W_shift_c2d_w155[48:55];
  assign mem_w_ck_c2d_w155[7]= W_shift_c2d_w155[56:63];
  assign mem_w_ck_c2d_w155[8]= W_shift_c2d_w155[64:71];
  
  assign mem_w_ck_c2d_w156[0]= W_shift_c2d_w156[0:7];
  assign mem_w_ck_c2d_w156[1]= W_shift_c2d_w156[8:15];
  assign mem_w_ck_c2d_w156[2]= W_shift_c2d_w156[16:23];
  assign mem_w_ck_c2d_w156[3]= W_shift_c2d_w156[24:31];
  assign mem_w_ck_c2d_w156[4]= W_shift_c2d_w156[32:39];
  assign mem_w_ck_c2d_w156[5]= W_shift_c2d_w156[40:47];
  assign mem_w_ck_c2d_w156[6]= W_shift_c2d_w156[48:55];
  assign mem_w_ck_c2d_w156[7]= W_shift_c2d_w156[56:63];
  assign mem_w_ck_c2d_w156[8]= W_shift_c2d_w156[64:71];
  
  assign mem_w_ck_c2d_w161[0]= W_shift_c2d_w161[0:7];
  assign mem_w_ck_c2d_w161[1]= W_shift_c2d_w161[8:15];
  assign mem_w_ck_c2d_w161[2]= W_shift_c2d_w161[16:23];
  assign mem_w_ck_c2d_w161[3]= W_shift_c2d_w161[24:31];
  assign mem_w_ck_c2d_w161[4]= W_shift_c2d_w161[32:39];
  assign mem_w_ck_c2d_w161[5]= W_shift_c2d_w161[40:47];
  assign mem_w_ck_c2d_w161[6]= W_shift_c2d_w161[48:55];
  assign mem_w_ck_c2d_w161[7]= W_shift_c2d_w161[56:63];
  assign mem_w_ck_c2d_w161[8]= W_shift_c2d_w161[64:71];
  
  assign mem_w_ck_c2d_w162[0]= W_shift_c2d_w162[0:7];
  assign mem_w_ck_c2d_w162[1]= W_shift_c2d_w162[8:15];
  assign mem_w_ck_c2d_w162[2]= W_shift_c2d_w162[16:23];
  assign mem_w_ck_c2d_w162[3]= W_shift_c2d_w162[24:31];
  assign mem_w_ck_c2d_w162[4]= W_shift_c2d_w162[32:39];
  assign mem_w_ck_c2d_w162[5]= W_shift_c2d_w162[40:47];
  assign mem_w_ck_c2d_w162[6]= W_shift_c2d_w162[48:55];
  assign mem_w_ck_c2d_w162[7]= W_shift_c2d_w162[56:63];
  assign mem_w_ck_c2d_w162[8]= W_shift_c2d_w162[64:71];
  
  assign mem_w_ck_c2d_w163[0]= W_shift_c2d_w163[0:7];
  assign mem_w_ck_c2d_w163[1]= W_shift_c2d_w163[8:15];
  assign mem_w_ck_c2d_w163[2]= W_shift_c2d_w163[16:23];
  assign mem_w_ck_c2d_w163[3]= W_shift_c2d_w163[24:31];
  assign mem_w_ck_c2d_w163[4]= W_shift_c2d_w163[32:39];
  assign mem_w_ck_c2d_w163[5]= W_shift_c2d_w163[40:47];
  assign mem_w_ck_c2d_w163[6]= W_shift_c2d_w163[48:55];
  assign mem_w_ck_c2d_w163[7]= W_shift_c2d_w163[56:63];
  assign mem_w_ck_c2d_w163[8]= W_shift_c2d_w163[64:71];
  
  assign mem_w_ck_c2d_w164[0]= W_shift_c2d_w164[0:7];
  assign mem_w_ck_c2d_w164[1]= W_shift_c2d_w164[8:15];
  assign mem_w_ck_c2d_w164[2]= W_shift_c2d_w164[16:23];
  assign mem_w_ck_c2d_w164[3]= W_shift_c2d_w164[24:31];
  assign mem_w_ck_c2d_w164[4]= W_shift_c2d_w164[32:39];
  assign mem_w_ck_c2d_w164[5]= W_shift_c2d_w164[40:47];
  assign mem_w_ck_c2d_w164[6]= W_shift_c2d_w164[48:55];
  assign mem_w_ck_c2d_w164[7]= W_shift_c2d_w164[56:63];
  assign mem_w_ck_c2d_w164[8]= W_shift_c2d_w164[64:71];
  
  assign mem_w_ck_c2d_w165[0]= W_shift_c2d_w165[0:7];
  assign mem_w_ck_c2d_w165[1]= W_shift_c2d_w165[8:15];
  assign mem_w_ck_c2d_w165[2]= W_shift_c2d_w165[16:23];
  assign mem_w_ck_c2d_w165[3]= W_shift_c2d_w165[24:31];
  assign mem_w_ck_c2d_w165[4]= W_shift_c2d_w165[32:39];
  assign mem_w_ck_c2d_w165[5]= W_shift_c2d_w165[40:47];
  assign mem_w_ck_c2d_w165[6]= W_shift_c2d_w165[48:55];
  assign mem_w_ck_c2d_w165[7]= W_shift_c2d_w165[56:63];
  assign mem_w_ck_c2d_w165[8]= W_shift_c2d_w165[64:71];
  
  assign mem_w_ck_c2d_w166[0]= W_shift_c2d_w166[0:7];
  assign mem_w_ck_c2d_w166[1]= W_shift_c2d_w166[8:15];
  assign mem_w_ck_c2d_w166[2]= W_shift_c2d_w166[16:23];
  assign mem_w_ck_c2d_w166[3]= W_shift_c2d_w166[24:31];
  assign mem_w_ck_c2d_w166[4]= W_shift_c2d_w166[32:39];
  assign mem_w_ck_c2d_w166[5]= W_shift_c2d_w166[40:47];
  assign mem_w_ck_c2d_w166[6]= W_shift_c2d_w166[48:55];
  assign mem_w_ck_c2d_w166[7]= W_shift_c2d_w166[56:63];
  assign mem_w_ck_c2d_w166[8]= W_shift_c2d_w166[64:71];
  
  always@(posedge (done_m_c2d_in1_w11 && done_m_c2d_in2_w12 && done_m_c2d_in3_w13 && done_m_c2d_in4_w14 && done_m_c2d_in5_w15 && done_m_c2d_in6_w16 && done_m_c2d_in1_w21 && done_m_c2d_in2_w22 && done_m_c2d_in3_w23 && done_m_c2d_in4_w24 && done_m_c2d_in5_w25 && done_m_c2d_in6_w26 && done_m_c2d_in1_w31 && done_m_c2d_in2_w32 && done_m_c2d_in3_w33 && done_m_c2d_in4_w34 && done_m_c2d_in5_w35 && done_m_c2d_in6_w36 && done_m_c2d_in1_w41 && done_m_c2d_in2_w42 && done_m_c2d_in3_w43 && done_m_c2d_in4_w44 && done_m_c2d_in5_w45 && done_m_c2d_in6_w46 && done_m_c2d_in1_w51 && done_m_c2d_in2_w52 && done_m_c2d_in3_w53 && done_m_c2d_in4_w54 && done_m_c2d_in5_w55 && done_m_c2d_in6_w56 && done_m_c2d_in1_w61 && done_m_c2d_in2_w62 && done_m_c2d_in3_w63 && done_m_c2d_in4_w64 && done_m_c2d_in5_w65 && done_m_c2d_in6_w66 && done_m_c2d_in1_w71 && done_m_c2d_in2_w72 && done_m_c2d_in3_w73 && done_m_c2d_in4_w74 && done_m_c2d_in5_w75 && done_m_c2d_in6_w76 && done_m_c2d_in1_w81 && done_m_c2d_in2_w82 && done_m_c2d_in3_w83 && done_m_c2d_in4_w84 && done_m_c2d_in5_w85 && done_m_c2d_in6_w86 && done_m_c2d_in1_w91 && done_m_c2d_in2_w92 && done_m_c2d_in3_w93 && done_m_c2d_in4_w94 && done_m_c2d_in5_w95 && done_m_c2d_in6_w96 && done_m_c2d_in1_w101 && done_m_c2d_in2_w102 && done_m_c2d_in3_w103 && done_m_c2d_in4_w104 && done_m_c2d_in5_w105 && done_m_c2d_in6_w106 && done_m_c2d_in1_w111 && done_m_c2d_in2_w112 && done_m_c2d_in3_w113 && done_m_c2d_in4_w114 && done_m_c2d_in5_w115 && done_m_c2d_in6_w116 && done_m_c2d_in1_w121 && done_m_c2d_in2_w122 && done_m_c2d_in3_w123 && done_m_c2d_in4_w124 && done_m_c2d_in5_w125 && done_m_c2d_in6_w126 && done_m_c2d_in1_w131 && done_m_c2d_in2_w132 && done_m_c2d_in3_w133 && done_m_c2d_in4_w134 && done_m_c2d_in5_w135 && done_m_c2d_in6_w136 && done_m_c2d_in1_w141 && done_m_c2d_in2_w142 && done_m_c2d_in3_w143 && done_m_c2d_in4_w144 && done_m_c2d_in5_w145 && done_m_c2d_in6_w146 && done_m_c2d_in1_w151 && done_m_c2d_in2_w152 && done_m_c2d_in3_w153 && done_m_c2d_in4_w154 && done_m_c2d_in5_w155 && done_m_c2d_in6_w156 && done_m_c2d_in1_w161 && done_m_c2d_in2_w162 && done_m_c2d_in3_w163 && done_m_c2d_in4_w164 && done_m_c2d_in5_w165 && done_m_c2d_in6_w166) or posedge rst_ck_c2d)begin
      if(rst_ck_c2d)begin
          count_ck_c2d=0;
          buffer_ck_c2d_in1_w11=0;
          buffer_ck_c2d_in2_w12=0;
          buffer_ck_c2d_in3_w13=0;
          buffer_ck_c2d_in4_w14=0;
          buffer_ck_c2d_in5_w15=0;
          buffer_ck_c2d_in6_w16=0;
          buffer_ck_c2d_in1_w21=0;
          buffer_ck_c2d_in2_w22=0;
          buffer_ck_c2d_in3_w23=0;
          buffer_ck_c2d_in4_w24=0;
          buffer_ck_c2d_in5_w25=0;
          buffer_ck_c2d_in6_w26=0;
          buffer_ck_c2d_in1_w31=0;
          buffer_ck_c2d_in2_w32=0;
          buffer_ck_c2d_in3_w33=0;
          buffer_ck_c2d_in4_w34=0;
          buffer_ck_c2d_in5_w35=0;
          buffer_ck_c2d_in6_w36=0;
          buffer_ck_c2d_in1_w41=0;
          buffer_ck_c2d_in2_w42=0;
          buffer_ck_c2d_in3_w43=0;
          buffer_ck_c2d_in4_w44=0;
          buffer_ck_c2d_in5_w45=0;
          buffer_ck_c2d_in6_w46=0;
          buffer_ck_c2d_in1_w51=0;
          buffer_ck_c2d_in2_w52=0;
          buffer_ck_c2d_in3_w53=0;
          buffer_ck_c2d_in4_w54=0;
          buffer_ck_c2d_in5_w55=0;
          buffer_ck_c2d_in6_w56=0;
          buffer_ck_c2d_in1_w61=0;
          buffer_ck_c2d_in2_w62=0;
          buffer_ck_c2d_in3_w63=0;
          buffer_ck_c2d_in4_w64=0;
          buffer_ck_c2d_in5_w65=0;
          buffer_ck_c2d_in6_w66=0;
          buffer_ck_c2d_in1_w71=0;
          buffer_ck_c2d_in2_w72=0;
          buffer_ck_c2d_in3_w73=0;
          buffer_ck_c2d_in4_w74=0;
          buffer_ck_c2d_in5_w75=0;
          buffer_ck_c2d_in6_w76=0;
          buffer_ck_c2d_in1_w81=0;
          buffer_ck_c2d_in2_w82=0;
          buffer_ck_c2d_in3_w83=0;
          buffer_ck_c2d_in4_w84=0;
          buffer_ck_c2d_in5_w85=0;
          buffer_ck_c2d_in6_w86=0;
          buffer_ck_c2d_in1_w91=0;
          buffer_ck_c2d_in2_w92=0;
          buffer_ck_c2d_in3_w93=0;
          buffer_ck_c2d_in4_w94=0;
          buffer_ck_c2d_in5_w95=0;
          buffer_ck_c2d_in6_w96=0;
          buffer_ck_c2d_in1_w101=0;
          buffer_ck_c2d_in2_w102=0;
          buffer_ck_c2d_in3_w103=0;
          buffer_ck_c2d_in4_w104=0;
          buffer_ck_c2d_in5_w105=0;
          buffer_ck_c2d_in6_w106=0;
          buffer_ck_c2d_in1_w111=0;
          buffer_ck_c2d_in2_w112=0;
          buffer_ck_c2d_in3_w113=0;
          buffer_ck_c2d_in4_w114=0;
          buffer_ck_c2d_in5_w115=0;
          buffer_ck_c2d_in6_w116=0;
          buffer_ck_c2d_in1_w121=0;
          buffer_ck_c2d_in2_w122=0;
          buffer_ck_c2d_in3_w123=0;
          buffer_ck_c2d_in4_w124=0;
          buffer_ck_c2d_in5_w125=0;
          buffer_ck_c2d_in6_w126=0;
          buffer_ck_c2d_in1_w131=0;
          buffer_ck_c2d_in2_w132=0;
          buffer_ck_c2d_in3_w133=0;
          buffer_ck_c2d_in4_w134=0;
          buffer_ck_c2d_in5_w135=0;
          buffer_ck_c2d_in6_w136=0;
          buffer_ck_c2d_in1_w141=0;
          buffer_ck_c2d_in2_w142=0;
          buffer_ck_c2d_in3_w143=0;
          buffer_ck_c2d_in4_w144=0;
          buffer_ck_c2d_in5_w145=0;
          buffer_ck_c2d_in6_w146=0;
          buffer_ck_c2d_in1_w151=0;
          buffer_ck_c2d_in2_w152=0;
          buffer_ck_c2d_in3_w153=0;
          buffer_ck_c2d_in4_w154=0;
          buffer_ck_c2d_in5_w155=0;
          buffer_ck_c2d_in6_w156=0;
          buffer_ck_c2d_in1_w161=0;
          buffer_ck_c2d_in2_w162=0;
          buffer_ck_c2d_in3_w163=0;
          buffer_ck_c2d_in4_w164=0;
          buffer_ck_c2d_in5_w165=0;
          buffer_ck_c2d_in6_w166=0;
      end
      else begin
          if(count_ck_c2d<9)begin
              count_ck_c2d=count_ck_c2d+1;
              buffer_ck_c2d_in1_w11=buffer_ck_c2d_in1_w11+result_temp_m_c2d_in1_w11;
              buffer_ck_c2d_in2_w12=buffer_ck_c2d_in2_w12+result_temp_m_c2d_in2_w12;
              buffer_ck_c2d_in3_w13=buffer_ck_c2d_in3_w13+result_temp_m_c2d_in3_w13;
              buffer_ck_c2d_in4_w14=buffer_ck_c2d_in4_w14+result_temp_m_c2d_in4_w14;
              buffer_ck_c2d_in5_w15=buffer_ck_c2d_in5_w15+result_temp_m_c2d_in5_w15;
              buffer_ck_c2d_in6_w16=buffer_ck_c2d_in6_w16+result_temp_m_c2d_in6_w16;
              buffer_ck_c2d_in1_w21=buffer_ck_c2d_in1_w21+result_temp_m_c2d_in1_w21;
              buffer_ck_c2d_in2_w22=buffer_ck_c2d_in2_w22+result_temp_m_c2d_in2_w22;
              buffer_ck_c2d_in3_w23=buffer_ck_c2d_in3_w23+result_temp_m_c2d_in3_w23;
              buffer_ck_c2d_in4_w24=buffer_ck_c2d_in4_w24+result_temp_m_c2d_in4_w24;
              buffer_ck_c2d_in5_w25=buffer_ck_c2d_in5_w25+result_temp_m_c2d_in5_w25;
              buffer_ck_c2d_in6_w26=buffer_ck_c2d_in6_w26+result_temp_m_c2d_in6_w26;
              buffer_ck_c2d_in1_w31=buffer_ck_c2d_in1_w31+result_temp_m_c2d_in1_w31;
              buffer_ck_c2d_in2_w32=buffer_ck_c2d_in2_w32+result_temp_m_c2d_in2_w32;
              buffer_ck_c2d_in3_w33=buffer_ck_c2d_in3_w33+result_temp_m_c2d_in3_w33;
              buffer_ck_c2d_in4_w34=buffer_ck_c2d_in4_w34+result_temp_m_c2d_in4_w34;
              buffer_ck_c2d_in5_w35=buffer_ck_c2d_in5_w35+result_temp_m_c2d_in5_w35;
              buffer_ck_c2d_in6_w36=buffer_ck_c2d_in6_w36+result_temp_m_c2d_in6_w36;
              buffer_ck_c2d_in1_w41=buffer_ck_c2d_in1_w41+result_temp_m_c2d_in1_w41;
              buffer_ck_c2d_in2_w42=buffer_ck_c2d_in2_w42+result_temp_m_c2d_in2_w42;
              buffer_ck_c2d_in3_w43=buffer_ck_c2d_in3_w43+result_temp_m_c2d_in3_w43;
              buffer_ck_c2d_in4_w44=buffer_ck_c2d_in4_w44+result_temp_m_c2d_in4_w44;
              buffer_ck_c2d_in5_w45=buffer_ck_c2d_in5_w45+result_temp_m_c2d_in5_w45;
              buffer_ck_c2d_in6_w46=buffer_ck_c2d_in6_w46+result_temp_m_c2d_in6_w46;
              buffer_ck_c2d_in1_w51=buffer_ck_c2d_in1_w51+result_temp_m_c2d_in1_w51;
              buffer_ck_c2d_in2_w52=buffer_ck_c2d_in2_w52+result_temp_m_c2d_in2_w52;
              buffer_ck_c2d_in3_w53=buffer_ck_c2d_in3_w53+result_temp_m_c2d_in3_w53;
              buffer_ck_c2d_in4_w54=buffer_ck_c2d_in4_w54+result_temp_m_c2d_in4_w54;
              buffer_ck_c2d_in5_w55=buffer_ck_c2d_in5_w55+result_temp_m_c2d_in5_w55;
              buffer_ck_c2d_in6_w56=buffer_ck_c2d_in6_w56+result_temp_m_c2d_in6_w56;
              buffer_ck_c2d_in1_w61=buffer_ck_c2d_in1_w61+result_temp_m_c2d_in1_w61;
              buffer_ck_c2d_in2_w62=buffer_ck_c2d_in2_w62+result_temp_m_c2d_in2_w62;
              buffer_ck_c2d_in3_w63=buffer_ck_c2d_in3_w63+result_temp_m_c2d_in3_w63;
              buffer_ck_c2d_in4_w64=buffer_ck_c2d_in4_w64+result_temp_m_c2d_in4_w64;
              buffer_ck_c2d_in5_w65=buffer_ck_c2d_in5_w65+result_temp_m_c2d_in5_w65;
              buffer_ck_c2d_in6_w66=buffer_ck_c2d_in6_w66+result_temp_m_c2d_in6_w66;
              buffer_ck_c2d_in1_w71=buffer_ck_c2d_in1_w71+result_temp_m_c2d_in1_w71;
              buffer_ck_c2d_in2_w72=buffer_ck_c2d_in2_w72+result_temp_m_c2d_in2_w72;
              buffer_ck_c2d_in3_w73=buffer_ck_c2d_in3_w73+result_temp_m_c2d_in3_w73;
              buffer_ck_c2d_in4_w74=buffer_ck_c2d_in4_w74+result_temp_m_c2d_in4_w74;
              buffer_ck_c2d_in5_w75=buffer_ck_c2d_in5_w75+result_temp_m_c2d_in5_w75;
              buffer_ck_c2d_in6_w76=buffer_ck_c2d_in6_w76+result_temp_m_c2d_in6_w76;
              buffer_ck_c2d_in1_w81=buffer_ck_c2d_in1_w81+result_temp_m_c2d_in1_w81;
              buffer_ck_c2d_in2_w82=buffer_ck_c2d_in2_w82+result_temp_m_c2d_in2_w82;
              buffer_ck_c2d_in3_w83=buffer_ck_c2d_in3_w83+result_temp_m_c2d_in3_w83;
              buffer_ck_c2d_in4_w84=buffer_ck_c2d_in4_w84+result_temp_m_c2d_in4_w84;
              buffer_ck_c2d_in5_w85=buffer_ck_c2d_in5_w85+result_temp_m_c2d_in5_w85;
              buffer_ck_c2d_in6_w86=buffer_ck_c2d_in6_w86+result_temp_m_c2d_in6_w86;
              buffer_ck_c2d_in1_w91=buffer_ck_c2d_in1_w91+result_temp_m_c2d_in1_w91;
              buffer_ck_c2d_in2_w92=buffer_ck_c2d_in2_w92+result_temp_m_c2d_in2_w92;
              buffer_ck_c2d_in3_w93=buffer_ck_c2d_in3_w93+result_temp_m_c2d_in3_w93;
              buffer_ck_c2d_in4_w94=buffer_ck_c2d_in4_w94+result_temp_m_c2d_in4_w94;
              buffer_ck_c2d_in5_w95=buffer_ck_c2d_in5_w95+result_temp_m_c2d_in5_w95;
              buffer_ck_c2d_in6_w96=buffer_ck_c2d_in6_w96+result_temp_m_c2d_in6_w96;
              buffer_ck_c2d_in1_w101=buffer_ck_c2d_in1_w101+result_temp_m_c2d_in1_w101;
              buffer_ck_c2d_in2_w102=buffer_ck_c2d_in2_w102+result_temp_m_c2d_in2_w102;
              buffer_ck_c2d_in3_w103=buffer_ck_c2d_in3_w103+result_temp_m_c2d_in3_w103;
              buffer_ck_c2d_in4_w104=buffer_ck_c2d_in4_w104+result_temp_m_c2d_in4_w104;
              buffer_ck_c2d_in5_w105=buffer_ck_c2d_in5_w105+result_temp_m_c2d_in5_w105;
              buffer_ck_c2d_in6_w106=buffer_ck_c2d_in6_w106+result_temp_m_c2d_in6_w106;
              buffer_ck_c2d_in1_w111=buffer_ck_c2d_in1_w111+result_temp_m_c2d_in1_w111;
              buffer_ck_c2d_in2_w112=buffer_ck_c2d_in2_w112+result_temp_m_c2d_in2_w112;
              buffer_ck_c2d_in3_w113=buffer_ck_c2d_in3_w113+result_temp_m_c2d_in3_w113;
              buffer_ck_c2d_in4_w114=buffer_ck_c2d_in4_w114+result_temp_m_c2d_in4_w114;
              buffer_ck_c2d_in5_w115=buffer_ck_c2d_in5_w115+result_temp_m_c2d_in5_w115;
              buffer_ck_c2d_in6_w116=buffer_ck_c2d_in6_w116+result_temp_m_c2d_in6_w116;
              buffer_ck_c2d_in1_w121=buffer_ck_c2d_in1_w121+result_temp_m_c2d_in1_w121;
              buffer_ck_c2d_in2_w122=buffer_ck_c2d_in2_w122+result_temp_m_c2d_in2_w122;
              buffer_ck_c2d_in3_w123=buffer_ck_c2d_in3_w123+result_temp_m_c2d_in3_w123;
              buffer_ck_c2d_in4_w124=buffer_ck_c2d_in4_w124+result_temp_m_c2d_in4_w124;
              buffer_ck_c2d_in5_w125=buffer_ck_c2d_in5_w125+result_temp_m_c2d_in5_w125;
              buffer_ck_c2d_in6_w126=buffer_ck_c2d_in6_w126+result_temp_m_c2d_in6_w126;
              buffer_ck_c2d_in1_w131=buffer_ck_c2d_in1_w131+result_temp_m_c2d_in1_w131;
              buffer_ck_c2d_in2_w132=buffer_ck_c2d_in2_w132+result_temp_m_c2d_in2_w132;
              buffer_ck_c2d_in3_w133=buffer_ck_c2d_in3_w133+result_temp_m_c2d_in3_w133;
              buffer_ck_c2d_in4_w134=buffer_ck_c2d_in4_w134+result_temp_m_c2d_in4_w134;
              buffer_ck_c2d_in5_w135=buffer_ck_c2d_in5_w135+result_temp_m_c2d_in5_w135;
              buffer_ck_c2d_in6_w136=buffer_ck_c2d_in6_w136+result_temp_m_c2d_in6_w136;
              buffer_ck_c2d_in1_w141=buffer_ck_c2d_in1_w141+result_temp_m_c2d_in1_w141;
              buffer_ck_c2d_in2_w142=buffer_ck_c2d_in2_w142+result_temp_m_c2d_in2_w142;
              buffer_ck_c2d_in3_w143=buffer_ck_c2d_in3_w143+result_temp_m_c2d_in3_w143;
              buffer_ck_c2d_in4_w144=buffer_ck_c2d_in4_w144+result_temp_m_c2d_in4_w144;
              buffer_ck_c2d_in5_w145=buffer_ck_c2d_in5_w145+result_temp_m_c2d_in5_w145;
              buffer_ck_c2d_in6_w146=buffer_ck_c2d_in6_w146+result_temp_m_c2d_in6_w146;
              buffer_ck_c2d_in1_w151=buffer_ck_c2d_in1_w151+result_temp_m_c2d_in1_w151;
              buffer_ck_c2d_in2_w152=buffer_ck_c2d_in2_w152+result_temp_m_c2d_in2_w152;
              buffer_ck_c2d_in3_w153=buffer_ck_c2d_in3_w153+result_temp_m_c2d_in3_w153;
              buffer_ck_c2d_in4_w154=buffer_ck_c2d_in4_w154+result_temp_m_c2d_in4_w154;
              buffer_ck_c2d_in5_w155=buffer_ck_c2d_in5_w155+result_temp_m_c2d_in5_w155;
              buffer_ck_c2d_in6_w156=buffer_ck_c2d_in6_w156+result_temp_m_c2d_in6_w156;
              buffer_ck_c2d_in1_w161=buffer_ck_c2d_in1_w161+result_temp_m_c2d_in1_w161;
              buffer_ck_c2d_in2_w162=buffer_ck_c2d_in2_w162+result_temp_m_c2d_in2_w162;
              buffer_ck_c2d_in3_w163=buffer_ck_c2d_in3_w163+result_temp_m_c2d_in3_w163;
              buffer_ck_c2d_in4_w164=buffer_ck_c2d_in4_w164+result_temp_m_c2d_in4_w164;
              buffer_ck_c2d_in5_w165=buffer_ck_c2d_in5_w165+result_temp_m_c2d_in5_w165;
              buffer_ck_c2d_in6_w166=buffer_ck_c2d_in6_w166+result_temp_m_c2d_in6_w166;
          end
          else begin
              count_ck_c2d=4'hX;
          end
      end
  end
  
  always@(*)begin
      case(state_ck_c2d)
          0:begin
              en_m_c2d=0;
              rst_m_c2d=1;
              x_in_m_c2d_ch_in1=0;
              x_in_m_c2d_ch_in2=0;
              x_in_m_c2d_ch_in3=0;
              x_in_m_c2d_ch_in4=0;
              x_in_m_c2d_ch_in5=0;
              x_in_m_c2d_ch_in6=0;
              w_in_m_c2d_w11=0;
              w_in_m_c2d_w12=0;
              w_in_m_c2d_w13=0;
              w_in_m_c2d_w14=0;
              w_in_m_c2d_w15=0;
              w_in_m_c2d_w16=0;
              w_in_m_c2d_w21=0;
              w_in_m_c2d_w22=0;
              w_in_m_c2d_w23=0;
              w_in_m_c2d_w24=0;
              w_in_m_c2d_w25=0;
              w_in_m_c2d_w26=0;
              w_in_m_c2d_w31=0;
              w_in_m_c2d_w32=0;
              w_in_m_c2d_w33=0;
              w_in_m_c2d_w34=0;
              w_in_m_c2d_w35=0;
              w_in_m_c2d_w36=0;
              w_in_m_c2d_w41=0;
              w_in_m_c2d_w42=0;
              w_in_m_c2d_w43=0;
              w_in_m_c2d_w44=0;
              w_in_m_c2d_w45=0;
              w_in_m_c2d_w46=0;
              w_in_m_c2d_w51=0;
              w_in_m_c2d_w52=0;
              w_in_m_c2d_w53=0;
              w_in_m_c2d_w54=0;
              w_in_m_c2d_w55=0;
              w_in_m_c2d_w56=0;
              w_in_m_c2d_w61=0;
              w_in_m_c2d_w62=0;
              w_in_m_c2d_w63=0;
              w_in_m_c2d_w64=0;
              w_in_m_c2d_w65=0;
              w_in_m_c2d_w66=0;
              w_in_m_c2d_w71=0;
              w_in_m_c2d_w72=0;
              w_in_m_c2d_w73=0;
              w_in_m_c2d_w74=0;
              w_in_m_c2d_w75=0;
              w_in_m_c2d_w76=0;
              w_in_m_c2d_w81=0;
              w_in_m_c2d_w82=0;
              w_in_m_c2d_w83=0;
              w_in_m_c2d_w84=0;
              w_in_m_c2d_w85=0;
              w_in_m_c2d_w86=0;
              w_in_m_c2d_w91=0;
              w_in_m_c2d_w92=0;
              w_in_m_c2d_w93=0;
              w_in_m_c2d_w94=0;
              w_in_m_c2d_w95=0;
              w_in_m_c2d_w96=0;
              w_in_m_c2d_w101=0;
              w_in_m_c2d_w102=0;
              w_in_m_c2d_w103=0;
              w_in_m_c2d_w104=0;
              w_in_m_c2d_w105=0;
              w_in_m_c2d_w106=0;
              w_in_m_c2d_w111=0;
              w_in_m_c2d_w112=0;
              w_in_m_c2d_w113=0;
              w_in_m_c2d_w114=0;
              w_in_m_c2d_w115=0;
              w_in_m_c2d_w116=0;
              w_in_m_c2d_w121=0;
              w_in_m_c2d_w122=0;
              w_in_m_c2d_w123=0;
              w_in_m_c2d_w124=0;
              w_in_m_c2d_w125=0;
              w_in_m_c2d_w126=0;
              w_in_m_c2d_w131=0;
              w_in_m_c2d_w132=0;
              w_in_m_c2d_w133=0;
              w_in_m_c2d_w134=0;
              w_in_m_c2d_w135=0;
              w_in_m_c2d_w136=0;
              w_in_m_c2d_w141=0;
              w_in_m_c2d_w142=0;
              w_in_m_c2d_w143=0;
              w_in_m_c2d_w144=0;
              w_in_m_c2d_w145=0;
              w_in_m_c2d_w146=0;
              w_in_m_c2d_w151=0;
              w_in_m_c2d_w152=0;
              w_in_m_c2d_w153=0;
              w_in_m_c2d_w154=0;
              w_in_m_c2d_w155=0;
              w_in_m_c2d_w156=0;
              w_in_m_c2d_w161=0;
              w_in_m_c2d_w162=0;
              w_in_m_c2d_w163=0;
              w_in_m_c2d_w164=0;
              w_in_m_c2d_w165=0;
              w_in_m_c2d_w166=0;
          end
          1: begin
              en_m_c2d=1;
              rst_m_c2d=0;
              x_in_m_c2d_ch_in1=mem_x_ck_c2d_ch_in1[count_ck_c2d];
              x_in_m_c2d_ch_in2=mem_x_ck_c2d_ch_in2[count_ck_c2d];
              x_in_m_c2d_ch_in3=mem_x_ck_c2d_ch_in3[count_ck_c2d];
              x_in_m_c2d_ch_in4=mem_x_ck_c2d_ch_in4[count_ck_c2d];
              x_in_m_c2d_ch_in5=mem_x_ck_c2d_ch_in5[count_ck_c2d];
              x_in_m_c2d_ch_in6=mem_x_ck_c2d_ch_in6[count_ck_c2d];
              w_in_m_c2d_w11=mem_w_ck_c2d_w11[count_ck_c2d];
              w_in_m_c2d_w12=mem_w_ck_c2d_w12[count_ck_c2d];
              w_in_m_c2d_w13=mem_w_ck_c2d_w13[count_ck_c2d];
              w_in_m_c2d_w14=mem_w_ck_c2d_w14[count_ck_c2d];
              w_in_m_c2d_w15=mem_w_ck_c2d_w15[count_ck_c2d];
              w_in_m_c2d_w16=mem_w_ck_c2d_w16[count_ck_c2d];
              w_in_m_c2d_w21=mem_w_ck_c2d_w21[count_ck_c2d];
              w_in_m_c2d_w22=mem_w_ck_c2d_w22[count_ck_c2d];
              w_in_m_c2d_w23=mem_w_ck_c2d_w23[count_ck_c2d];
              w_in_m_c2d_w24=mem_w_ck_c2d_w24[count_ck_c2d];
              w_in_m_c2d_w25=mem_w_ck_c2d_w25[count_ck_c2d];
              w_in_m_c2d_w26=mem_w_ck_c2d_w26[count_ck_c2d];
              w_in_m_c2d_w31=mem_w_ck_c2d_w31[count_ck_c2d];
              w_in_m_c2d_w32=mem_w_ck_c2d_w32[count_ck_c2d];
              w_in_m_c2d_w33=mem_w_ck_c2d_w33[count_ck_c2d];
              w_in_m_c2d_w34=mem_w_ck_c2d_w34[count_ck_c2d];
              w_in_m_c2d_w35=mem_w_ck_c2d_w35[count_ck_c2d];
              w_in_m_c2d_w36=mem_w_ck_c2d_w36[count_ck_c2d];
              w_in_m_c2d_w41=mem_w_ck_c2d_w41[count_ck_c2d];
              w_in_m_c2d_w42=mem_w_ck_c2d_w42[count_ck_c2d];
              w_in_m_c2d_w43=mem_w_ck_c2d_w43[count_ck_c2d];
              w_in_m_c2d_w44=mem_w_ck_c2d_w44[count_ck_c2d];
              w_in_m_c2d_w45=mem_w_ck_c2d_w45[count_ck_c2d];
              w_in_m_c2d_w46=mem_w_ck_c2d_w46[count_ck_c2d];
              w_in_m_c2d_w51=mem_w_ck_c2d_w51[count_ck_c2d];
              w_in_m_c2d_w52=mem_w_ck_c2d_w52[count_ck_c2d];
              w_in_m_c2d_w53=mem_w_ck_c2d_w53[count_ck_c2d];
              w_in_m_c2d_w54=mem_w_ck_c2d_w54[count_ck_c2d];
              w_in_m_c2d_w55=mem_w_ck_c2d_w55[count_ck_c2d];
              w_in_m_c2d_w56=mem_w_ck_c2d_w56[count_ck_c2d];
              w_in_m_c2d_w61=mem_w_ck_c2d_w61[count_ck_c2d];
              w_in_m_c2d_w62=mem_w_ck_c2d_w62[count_ck_c2d];
              w_in_m_c2d_w63=mem_w_ck_c2d_w63[count_ck_c2d];
              w_in_m_c2d_w64=mem_w_ck_c2d_w64[count_ck_c2d];
              w_in_m_c2d_w65=mem_w_ck_c2d_w65[count_ck_c2d];
              w_in_m_c2d_w66=mem_w_ck_c2d_w66[count_ck_c2d];
              w_in_m_c2d_w71=mem_w_ck_c2d_w71[count_ck_c2d];
              w_in_m_c2d_w72=mem_w_ck_c2d_w72[count_ck_c2d];
              w_in_m_c2d_w73=mem_w_ck_c2d_w73[count_ck_c2d];
              w_in_m_c2d_w74=mem_w_ck_c2d_w74[count_ck_c2d];
              w_in_m_c2d_w75=mem_w_ck_c2d_w75[count_ck_c2d];
              w_in_m_c2d_w76=mem_w_ck_c2d_w76[count_ck_c2d];
              w_in_m_c2d_w81=mem_w_ck_c2d_w81[count_ck_c2d];
              w_in_m_c2d_w82=mem_w_ck_c2d_w82[count_ck_c2d];
              w_in_m_c2d_w83=mem_w_ck_c2d_w83[count_ck_c2d];
              w_in_m_c2d_w84=mem_w_ck_c2d_w84[count_ck_c2d];
              w_in_m_c2d_w85=mem_w_ck_c2d_w85[count_ck_c2d];
              w_in_m_c2d_w86=mem_w_ck_c2d_w86[count_ck_c2d];
              w_in_m_c2d_w91=mem_w_ck_c2d_w91[count_ck_c2d];
              w_in_m_c2d_w92=mem_w_ck_c2d_w92[count_ck_c2d];
              w_in_m_c2d_w93=mem_w_ck_c2d_w93[count_ck_c2d];
              w_in_m_c2d_w94=mem_w_ck_c2d_w94[count_ck_c2d];
              w_in_m_c2d_w95=mem_w_ck_c2d_w95[count_ck_c2d];
              w_in_m_c2d_w96=mem_w_ck_c2d_w96[count_ck_c2d];
              w_in_m_c2d_w101=mem_w_ck_c2d_w101[count_ck_c2d];
              w_in_m_c2d_w102=mem_w_ck_c2d_w102[count_ck_c2d];
              w_in_m_c2d_w103=mem_w_ck_c2d_w103[count_ck_c2d];
              w_in_m_c2d_w104=mem_w_ck_c2d_w104[count_ck_c2d];
              w_in_m_c2d_w105=mem_w_ck_c2d_w105[count_ck_c2d];
              w_in_m_c2d_w106=mem_w_ck_c2d_w106[count_ck_c2d];
              w_in_m_c2d_w111=mem_w_ck_c2d_w111[count_ck_c2d];
              w_in_m_c2d_w112=mem_w_ck_c2d_w112[count_ck_c2d];
              w_in_m_c2d_w113=mem_w_ck_c2d_w113[count_ck_c2d];
              w_in_m_c2d_w114=mem_w_ck_c2d_w114[count_ck_c2d];
              w_in_m_c2d_w115=mem_w_ck_c2d_w115[count_ck_c2d];
              w_in_m_c2d_w116=mem_w_ck_c2d_w116[count_ck_c2d];
              w_in_m_c2d_w121=mem_w_ck_c2d_w121[count_ck_c2d];
              w_in_m_c2d_w122=mem_w_ck_c2d_w122[count_ck_c2d];
              w_in_m_c2d_w123=mem_w_ck_c2d_w123[count_ck_c2d];
              w_in_m_c2d_w124=mem_w_ck_c2d_w124[count_ck_c2d];
              w_in_m_c2d_w125=mem_w_ck_c2d_w125[count_ck_c2d];
              w_in_m_c2d_w126=mem_w_ck_c2d_w126[count_ck_c2d];
              w_in_m_c2d_w131=mem_w_ck_c2d_w131[count_ck_c2d];
              w_in_m_c2d_w132=mem_w_ck_c2d_w132[count_ck_c2d];
              w_in_m_c2d_w133=mem_w_ck_c2d_w133[count_ck_c2d];
              w_in_m_c2d_w134=mem_w_ck_c2d_w134[count_ck_c2d];
              w_in_m_c2d_w135=mem_w_ck_c2d_w135[count_ck_c2d];
              w_in_m_c2d_w136=mem_w_ck_c2d_w136[count_ck_c2d];
              w_in_m_c2d_w141=mem_w_ck_c2d_w141[count_ck_c2d];
              w_in_m_c2d_w142=mem_w_ck_c2d_w142[count_ck_c2d];
              w_in_m_c2d_w143=mem_w_ck_c2d_w143[count_ck_c2d];
              w_in_m_c2d_w144=mem_w_ck_c2d_w144[count_ck_c2d];
              w_in_m_c2d_w145=mem_w_ck_c2d_w145[count_ck_c2d];
              w_in_m_c2d_w146=mem_w_ck_c2d_w146[count_ck_c2d];
              w_in_m_c2d_w151=mem_w_ck_c2d_w151[count_ck_c2d];
              w_in_m_c2d_w152=mem_w_ck_c2d_w152[count_ck_c2d];
              w_in_m_c2d_w153=mem_w_ck_c2d_w153[count_ck_c2d];
              w_in_m_c2d_w154=mem_w_ck_c2d_w154[count_ck_c2d];
              w_in_m_c2d_w155=mem_w_ck_c2d_w155[count_ck_c2d];
              w_in_m_c2d_w156=mem_w_ck_c2d_w156[count_ck_c2d];
              w_in_m_c2d_w161=mem_w_ck_c2d_w161[count_ck_c2d];
              w_in_m_c2d_w162=mem_w_ck_c2d_w162[count_ck_c2d];
              w_in_m_c2d_w163=mem_w_ck_c2d_w163[count_ck_c2d];
              w_in_m_c2d_w164=mem_w_ck_c2d_w164[count_ck_c2d];
              w_in_m_c2d_w165=mem_w_ck_c2d_w165[count_ck_c2d];
              w_in_m_c2d_w166=mem_w_ck_c2d_w166[count_ck_c2d];
          end
          2:begin
              en_m_c2d=0;
              rst_m_c2d=1;
              x_in_m_c2d_ch_in1=0;
              x_in_m_c2d_ch_in2=0;
              x_in_m_c2d_ch_in3=0;
              x_in_m_c2d_ch_in4=0;
              x_in_m_c2d_ch_in5=0;
              x_in_m_c2d_ch_in6=0;
              w_in_m_c2d_w11=0;
              w_in_m_c2d_w12=0;
              w_in_m_c2d_w13=0;
              w_in_m_c2d_w14=0;
              w_in_m_c2d_w15=0;
              w_in_m_c2d_w16=0;
              w_in_m_c2d_w21=0;
              w_in_m_c2d_w22=0;
              w_in_m_c2d_w23=0;
              w_in_m_c2d_w24=0;
              w_in_m_c2d_w25=0;
              w_in_m_c2d_w26=0;
              w_in_m_c2d_w31=0;
              w_in_m_c2d_w32=0;
              w_in_m_c2d_w33=0;
              w_in_m_c2d_w34=0;
              w_in_m_c2d_w35=0;
              w_in_m_c2d_w36=0;
              w_in_m_c2d_w41=0;
              w_in_m_c2d_w42=0;
              w_in_m_c2d_w43=0;
              w_in_m_c2d_w44=0;
              w_in_m_c2d_w45=0;
              w_in_m_c2d_w46=0;
              w_in_m_c2d_w51=0;
              w_in_m_c2d_w52=0;
              w_in_m_c2d_w53=0;
              w_in_m_c2d_w54=0;
              w_in_m_c2d_w55=0;
              w_in_m_c2d_w56=0;
              w_in_m_c2d_w61=0;
              w_in_m_c2d_w62=0;
              w_in_m_c2d_w63=0;
              w_in_m_c2d_w64=0;
              w_in_m_c2d_w65=0;
              w_in_m_c2d_w66=0;
              w_in_m_c2d_w71=0;
              w_in_m_c2d_w72=0;
              w_in_m_c2d_w73=0;
              w_in_m_c2d_w74=0;
              w_in_m_c2d_w75=0;
              w_in_m_c2d_w76=0;
              w_in_m_c2d_w81=0;
              w_in_m_c2d_w82=0;
              w_in_m_c2d_w83=0;
              w_in_m_c2d_w84=0;
              w_in_m_c2d_w85=0;
              w_in_m_c2d_w86=0;
              w_in_m_c2d_w91=0;
              w_in_m_c2d_w92=0;
              w_in_m_c2d_w93=0;
              w_in_m_c2d_w94=0;
              w_in_m_c2d_w95=0;
              w_in_m_c2d_w96=0;
              w_in_m_c2d_w101=0;
              w_in_m_c2d_w102=0;
              w_in_m_c2d_w103=0;
              w_in_m_c2d_w104=0;
              w_in_m_c2d_w105=0;
              w_in_m_c2d_w106=0;
              w_in_m_c2d_w111=0;
              w_in_m_c2d_w112=0;
              w_in_m_c2d_w113=0;
              w_in_m_c2d_w114=0;
              w_in_m_c2d_w115=0;
              w_in_m_c2d_w116=0;
              w_in_m_c2d_w121=0;
              w_in_m_c2d_w122=0;
              w_in_m_c2d_w123=0;
              w_in_m_c2d_w124=0;
              w_in_m_c2d_w125=0;
              w_in_m_c2d_w126=0;
              w_in_m_c2d_w131=0;
              w_in_m_c2d_w132=0;
              w_in_m_c2d_w133=0;
              w_in_m_c2d_w134=0;
              w_in_m_c2d_w135=0;
              w_in_m_c2d_w136=0;
              w_in_m_c2d_w141=0;
              w_in_m_c2d_w142=0;
              w_in_m_c2d_w143=0;
              w_in_m_c2d_w144=0;
              w_in_m_c2d_w145=0;
              w_in_m_c2d_w146=0;
              w_in_m_c2d_w151=0;
              w_in_m_c2d_w152=0;
              w_in_m_c2d_w153=0;
              w_in_m_c2d_w154=0;
              w_in_m_c2d_w155=0;
              w_in_m_c2d_w156=0;
              w_in_m_c2d_w161=0;
              w_in_m_c2d_w162=0;
              w_in_m_c2d_w163=0;
              w_in_m_c2d_w164=0;
              w_in_m_c2d_w165=0;
              w_in_m_c2d_w166=0;
          end
          default:begin
              en_m_c2d=0;
              rst_m_c2d=1;
              x_in_m_c2d_ch_in1=0;
              x_in_m_c2d_ch_in2=0;
              x_in_m_c2d_ch_in3=0;
              x_in_m_c2d_ch_in4=0;
              x_in_m_c2d_ch_in5=0;
              x_in_m_c2d_ch_in6=0;
              w_in_m_c2d_w11=0;
              w_in_m_c2d_w12=0;
              w_in_m_c2d_w13=0;
              w_in_m_c2d_w14=0;
              w_in_m_c2d_w15=0;
              w_in_m_c2d_w16=0;
              w_in_m_c2d_w21=0;
              w_in_m_c2d_w22=0;
              w_in_m_c2d_w23=0;
              w_in_m_c2d_w24=0;
              w_in_m_c2d_w25=0;
              w_in_m_c2d_w26=0;
              w_in_m_c2d_w31=0;
              w_in_m_c2d_w32=0;
              w_in_m_c2d_w33=0;
              w_in_m_c2d_w34=0;
              w_in_m_c2d_w35=0;
              w_in_m_c2d_w36=0;
              w_in_m_c2d_w41=0;
              w_in_m_c2d_w42=0;
              w_in_m_c2d_w43=0;
              w_in_m_c2d_w44=0;
              w_in_m_c2d_w45=0;
              w_in_m_c2d_w46=0;
              w_in_m_c2d_w51=0;
              w_in_m_c2d_w52=0;
              w_in_m_c2d_w53=0;
              w_in_m_c2d_w54=0;
              w_in_m_c2d_w55=0;
              w_in_m_c2d_w56=0;
              w_in_m_c2d_w61=0;
              w_in_m_c2d_w62=0;
              w_in_m_c2d_w63=0;
              w_in_m_c2d_w64=0;
              w_in_m_c2d_w65=0;
              w_in_m_c2d_w66=0;
              w_in_m_c2d_w71=0;
              w_in_m_c2d_w72=0;
              w_in_m_c2d_w73=0;
              w_in_m_c2d_w74=0;
              w_in_m_c2d_w75=0;
              w_in_m_c2d_w76=0;
              w_in_m_c2d_w81=0;
              w_in_m_c2d_w82=0;
              w_in_m_c2d_w83=0;
              w_in_m_c2d_w84=0;
              w_in_m_c2d_w85=0;
              w_in_m_c2d_w86=0;
              w_in_m_c2d_w91=0;
              w_in_m_c2d_w92=0;
              w_in_m_c2d_w93=0;
              w_in_m_c2d_w94=0;
              w_in_m_c2d_w95=0;
              w_in_m_c2d_w96=0;
              w_in_m_c2d_w101=0;
              w_in_m_c2d_w102=0;
              w_in_m_c2d_w103=0;
              w_in_m_c2d_w104=0;
              w_in_m_c2d_w105=0;
              w_in_m_c2d_w106=0;
              w_in_m_c2d_w111=0;
              w_in_m_c2d_w112=0;
              w_in_m_c2d_w113=0;
              w_in_m_c2d_w114=0;
              w_in_m_c2d_w115=0;
              w_in_m_c2d_w116=0;
              w_in_m_c2d_w121=0;
              w_in_m_c2d_w122=0;
              w_in_m_c2d_w123=0;
              w_in_m_c2d_w124=0;
              w_in_m_c2d_w125=0;
              w_in_m_c2d_w126=0;
              w_in_m_c2d_w131=0;
              w_in_m_c2d_w132=0;
              w_in_m_c2d_w133=0;
              w_in_m_c2d_w134=0;
              w_in_m_c2d_w135=0;
              w_in_m_c2d_w136=0;
              w_in_m_c2d_w141=0;
              w_in_m_c2d_w142=0;
              w_in_m_c2d_w143=0;
              w_in_m_c2d_w144=0;
              w_in_m_c2d_w145=0;
              w_in_m_c2d_w146=0;
              w_in_m_c2d_w151=0;
              w_in_m_c2d_w152=0;
              w_in_m_c2d_w153=0;
              w_in_m_c2d_w154=0;
              w_in_m_c2d_w155=0;
              w_in_m_c2d_w156=0;
              w_in_m_c2d_w161=0;
              w_in_m_c2d_w162=0;
              w_in_m_c2d_w163=0;
              w_in_m_c2d_w164=0;
              w_in_m_c2d_w165=0;
              w_in_m_c2d_w166=0;
          end
      endcase
      if(count_ck_c2d==9)begin
          result_final_temp_ck_c2d_in1_w11=buffer_ck_c2d_in1_w11>>>4;
          result_final_temp_ck_c2d_in2_w12=buffer_ck_c2d_in2_w12>>>4;
          result_final_temp_ck_c2d_in3_w13=buffer_ck_c2d_in3_w13>>>4;
          result_final_temp_ck_c2d_in4_w14=buffer_ck_c2d_in4_w14>>>4;
          result_final_temp_ck_c2d_in5_w15=buffer_ck_c2d_in5_w15>>>4;
          result_final_temp_ck_c2d_in6_w16=buffer_ck_c2d_in6_w16>>>4;
          result_final_temp_ck_c2d_in1_w21=buffer_ck_c2d_in1_w21>>>4;
          result_final_temp_ck_c2d_in2_w22=buffer_ck_c2d_in2_w22>>>4;
          result_final_temp_ck_c2d_in3_w23=buffer_ck_c2d_in3_w23>>>4;
          result_final_temp_ck_c2d_in4_w24=buffer_ck_c2d_in4_w24>>>4;
          result_final_temp_ck_c2d_in5_w25=buffer_ck_c2d_in5_w25>>>4;
          result_final_temp_ck_c2d_in6_w26=buffer_ck_c2d_in6_w26>>>4;
          result_final_temp_ck_c2d_in1_w31=buffer_ck_c2d_in1_w31>>>4;
          result_final_temp_ck_c2d_in2_w32=buffer_ck_c2d_in2_w32>>>4;
          result_final_temp_ck_c2d_in3_w33=buffer_ck_c2d_in3_w33>>>4;
          result_final_temp_ck_c2d_in4_w34=buffer_ck_c2d_in4_w34>>>4;
          result_final_temp_ck_c2d_in5_w35=buffer_ck_c2d_in5_w35>>>4;
          result_final_temp_ck_c2d_in6_w36=buffer_ck_c2d_in6_w36>>>4;
          result_final_temp_ck_c2d_in1_w41=buffer_ck_c2d_in1_w41>>>4;
          result_final_temp_ck_c2d_in2_w42=buffer_ck_c2d_in2_w42>>>4;
          result_final_temp_ck_c2d_in3_w43=buffer_ck_c2d_in3_w43>>>4;
          result_final_temp_ck_c2d_in4_w44=buffer_ck_c2d_in4_w44>>>4;
          result_final_temp_ck_c2d_in5_w45=buffer_ck_c2d_in5_w45>>>4;
          result_final_temp_ck_c2d_in6_w46=buffer_ck_c2d_in6_w46>>>4;
          result_final_temp_ck_c2d_in1_w51=buffer_ck_c2d_in1_w51>>>4;
          result_final_temp_ck_c2d_in2_w52=buffer_ck_c2d_in2_w52>>>4;
          result_final_temp_ck_c2d_in3_w53=buffer_ck_c2d_in3_w53>>>4;
          result_final_temp_ck_c2d_in4_w54=buffer_ck_c2d_in4_w54>>>4;
          result_final_temp_ck_c2d_in5_w55=buffer_ck_c2d_in5_w55>>>4;
          result_final_temp_ck_c2d_in6_w56=buffer_ck_c2d_in6_w56>>>4;
          result_final_temp_ck_c2d_in1_w61=buffer_ck_c2d_in1_w61>>>4;
          result_final_temp_ck_c2d_in2_w62=buffer_ck_c2d_in2_w62>>>4;
          result_final_temp_ck_c2d_in3_w63=buffer_ck_c2d_in3_w63>>>4;
          result_final_temp_ck_c2d_in4_w64=buffer_ck_c2d_in4_w64>>>4;
          result_final_temp_ck_c2d_in5_w65=buffer_ck_c2d_in5_w65>>>4;
          result_final_temp_ck_c2d_in6_w66=buffer_ck_c2d_in6_w66>>>4;
          result_final_temp_ck_c2d_in1_w71=buffer_ck_c2d_in1_w71>>>4;
          result_final_temp_ck_c2d_in2_w72=buffer_ck_c2d_in2_w72>>>4;
          result_final_temp_ck_c2d_in3_w73=buffer_ck_c2d_in3_w73>>>4;
          result_final_temp_ck_c2d_in4_w74=buffer_ck_c2d_in4_w74>>>4;
          result_final_temp_ck_c2d_in5_w75=buffer_ck_c2d_in5_w75>>>4;
          result_final_temp_ck_c2d_in6_w76=buffer_ck_c2d_in6_w76>>>4;
          result_final_temp_ck_c2d_in1_w81=buffer_ck_c2d_in1_w81>>>4;
          result_final_temp_ck_c2d_in2_w82=buffer_ck_c2d_in2_w82>>>4;
          result_final_temp_ck_c2d_in3_w83=buffer_ck_c2d_in3_w83>>>4;
          result_final_temp_ck_c2d_in4_w84=buffer_ck_c2d_in4_w84>>>4;
          result_final_temp_ck_c2d_in5_w85=buffer_ck_c2d_in5_w85>>>4;
          result_final_temp_ck_c2d_in6_w86=buffer_ck_c2d_in6_w86>>>4;
          result_final_temp_ck_c2d_in1_w91=buffer_ck_c2d_in1_w91>>>4;
          result_final_temp_ck_c2d_in2_w92=buffer_ck_c2d_in2_w92>>>4;
          result_final_temp_ck_c2d_in3_w93=buffer_ck_c2d_in3_w93>>>4;
          result_final_temp_ck_c2d_in4_w94=buffer_ck_c2d_in4_w94>>>4;
          result_final_temp_ck_c2d_in5_w95=buffer_ck_c2d_in5_w95>>>4;
          result_final_temp_ck_c2d_in6_w96=buffer_ck_c2d_in6_w96>>>4;
          result_final_temp_ck_c2d_in1_w101=buffer_ck_c2d_in1_w101>>>4;
          result_final_temp_ck_c2d_in2_w102=buffer_ck_c2d_in2_w102>>>4;
          result_final_temp_ck_c2d_in3_w103=buffer_ck_c2d_in3_w103>>>4;
          result_final_temp_ck_c2d_in4_w104=buffer_ck_c2d_in4_w104>>>4;
          result_final_temp_ck_c2d_in5_w105=buffer_ck_c2d_in5_w105>>>4;
          result_final_temp_ck_c2d_in6_w106=buffer_ck_c2d_in6_w106>>>4;
          result_final_temp_ck_c2d_in1_w111=buffer_ck_c2d_in1_w111>>>4;
          result_final_temp_ck_c2d_in2_w112=buffer_ck_c2d_in2_w112>>>4;
          result_final_temp_ck_c2d_in3_w113=buffer_ck_c2d_in3_w113>>>4;
          result_final_temp_ck_c2d_in4_w114=buffer_ck_c2d_in4_w114>>>4;
          result_final_temp_ck_c2d_in5_w115=buffer_ck_c2d_in5_w115>>>4;
          result_final_temp_ck_c2d_in6_w116=buffer_ck_c2d_in6_w116>>>4;
          result_final_temp_ck_c2d_in1_w121=buffer_ck_c2d_in1_w121>>>4;
          result_final_temp_ck_c2d_in2_w122=buffer_ck_c2d_in2_w122>>>4;
          result_final_temp_ck_c2d_in3_w123=buffer_ck_c2d_in3_w123>>>4;
          result_final_temp_ck_c2d_in4_w124=buffer_ck_c2d_in4_w124>>>4;
          result_final_temp_ck_c2d_in5_w125=buffer_ck_c2d_in5_w125>>>4;
          result_final_temp_ck_c2d_in6_w126=buffer_ck_c2d_in6_w126>>>4;
          result_final_temp_ck_c2d_in1_w131=buffer_ck_c2d_in1_w131>>>4;
          result_final_temp_ck_c2d_in2_w132=buffer_ck_c2d_in2_w132>>>4;
          result_final_temp_ck_c2d_in3_w133=buffer_ck_c2d_in3_w133>>>4;
          result_final_temp_ck_c2d_in4_w134=buffer_ck_c2d_in4_w134>>>4;
          result_final_temp_ck_c2d_in5_w135=buffer_ck_c2d_in5_w135>>>4;
          result_final_temp_ck_c2d_in6_w136=buffer_ck_c2d_in6_w136>>>4;
          result_final_temp_ck_c2d_in1_w141=buffer_ck_c2d_in1_w141>>>4;
          result_final_temp_ck_c2d_in2_w142=buffer_ck_c2d_in2_w142>>>4;
          result_final_temp_ck_c2d_in3_w143=buffer_ck_c2d_in3_w143>>>4;
          result_final_temp_ck_c2d_in4_w144=buffer_ck_c2d_in4_w144>>>4;
          result_final_temp_ck_c2d_in5_w145=buffer_ck_c2d_in5_w145>>>4;
          result_final_temp_ck_c2d_in6_w146=buffer_ck_c2d_in6_w146>>>4;
          result_final_temp_ck_c2d_in1_w151=buffer_ck_c2d_in1_w151>>>4;
          result_final_temp_ck_c2d_in2_w152=buffer_ck_c2d_in2_w152>>>4;
          result_final_temp_ck_c2d_in3_w153=buffer_ck_c2d_in3_w153>>>4;
          result_final_temp_ck_c2d_in4_w154=buffer_ck_c2d_in4_w154>>>4;
          result_final_temp_ck_c2d_in5_w155=buffer_ck_c2d_in5_w155>>>4;
          result_final_temp_ck_c2d_in6_w156=buffer_ck_c2d_in6_w156>>>4;
          result_final_temp_ck_c2d_in1_w161=buffer_ck_c2d_in1_w161>>>4;
          result_final_temp_ck_c2d_in2_w162=buffer_ck_c2d_in2_w162>>>4;
          result_final_temp_ck_c2d_in3_w163=buffer_ck_c2d_in3_w163>>>4;
          result_final_temp_ck_c2d_in4_w164=buffer_ck_c2d_in4_w164>>>4;
          result_final_temp_ck_c2d_in5_w165=buffer_ck_c2d_in5_w165>>>4;
          result_final_temp_ck_c2d_in6_w166=buffer_ck_c2d_in6_w166>>>4;
  
          if(result_final_temp_ck_c2d_in1_w11>32767 || result_final_temp_ck_c2d_in1_w11<-32768)begin
              if(result_final_temp_ck_c2d_in1_w11>32767)
                  result_temp_ck_c2d_in1_w11 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w11 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w11 = result_final_temp_ck_c2d_in1_w11[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w12>32767 || result_final_temp_ck_c2d_in2_w12<-32768)begin
              if(result_final_temp_ck_c2d_in2_w12>32767)
                  result_temp_ck_c2d_in2_w12 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w12 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w12 = result_final_temp_ck_c2d_in2_w12[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w13>32767 || result_final_temp_ck_c2d_in3_w13<-32768)begin
              if(result_final_temp_ck_c2d_in3_w13>32767)
                  result_temp_ck_c2d_in3_w13 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w13 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w13 = result_final_temp_ck_c2d_in3_w13[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w14>32767 || result_final_temp_ck_c2d_in4_w14<-32768)begin
              if(result_final_temp_ck_c2d_in4_w14>32767)
                  result_temp_ck_c2d_in4_w14 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w14 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w14 = result_final_temp_ck_c2d_in4_w14[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w15>32767 || result_final_temp_ck_c2d_in5_w15<-32768)begin
              if(result_final_temp_ck_c2d_in5_w15>32767)
                  result_temp_ck_c2d_in5_w15 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w15 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w15 = result_final_temp_ck_c2d_in5_w15[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w16>32767 || result_final_temp_ck_c2d_in6_w16<-32768)begin
              if(result_final_temp_ck_c2d_in6_w16>32767)
                  result_temp_ck_c2d_in6_w16 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w16 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w16 = result_final_temp_ck_c2d_in6_w16[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w21>32767 || result_final_temp_ck_c2d_in1_w21<-32768)begin
              if(result_final_temp_ck_c2d_in1_w21>32767)
                  result_temp_ck_c2d_in1_w21 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w21 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w21 = result_final_temp_ck_c2d_in1_w21[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w22>32767 || result_final_temp_ck_c2d_in2_w22<-32768)begin
              if(result_final_temp_ck_c2d_in2_w22>32767)
                  result_temp_ck_c2d_in2_w22 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w22 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w22 = result_final_temp_ck_c2d_in2_w22[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w23>32767 || result_final_temp_ck_c2d_in3_w23<-32768)begin
              if(result_final_temp_ck_c2d_in3_w23>32767)
                  result_temp_ck_c2d_in3_w23 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w23 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w23 = result_final_temp_ck_c2d_in3_w23[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w24>32767 || result_final_temp_ck_c2d_in4_w24<-32768)begin
              if(result_final_temp_ck_c2d_in4_w24>32767)
                  result_temp_ck_c2d_in4_w24 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w24 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w24 = result_final_temp_ck_c2d_in4_w24[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w25>32767 || result_final_temp_ck_c2d_in5_w25<-32768)begin
              if(result_final_temp_ck_c2d_in5_w25>32767)
                  result_temp_ck_c2d_in5_w25 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w25 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w25 = result_final_temp_ck_c2d_in5_w25[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w26>32767 || result_final_temp_ck_c2d_in6_w26<-32768)begin
              if(result_final_temp_ck_c2d_in6_w26>32767)
                  result_temp_ck_c2d_in6_w26 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w26 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w26 = result_final_temp_ck_c2d_in6_w26[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w31>32767 || result_final_temp_ck_c2d_in1_w31<-32768)begin
              if(result_final_temp_ck_c2d_in1_w31>32767)
                  result_temp_ck_c2d_in1_w31 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w31 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w31 = result_final_temp_ck_c2d_in1_w31[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w32>32767 || result_final_temp_ck_c2d_in2_w32<-32768)begin
              if(result_final_temp_ck_c2d_in2_w32>32767)
                  result_temp_ck_c2d_in2_w32 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w32 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w32 = result_final_temp_ck_c2d_in2_w32[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w33>32767 || result_final_temp_ck_c2d_in3_w33<-32768)begin
              if(result_final_temp_ck_c2d_in3_w33>32767)
                  result_temp_ck_c2d_in3_w33 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w33 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w33 = result_final_temp_ck_c2d_in3_w33[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w34>32767 || result_final_temp_ck_c2d_in4_w34<-32768)begin
              if(result_final_temp_ck_c2d_in4_w34>32767)
                  result_temp_ck_c2d_in4_w34 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w34 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w34 = result_final_temp_ck_c2d_in4_w34[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w35>32767 || result_final_temp_ck_c2d_in5_w35<-32768)begin
              if(result_final_temp_ck_c2d_in5_w35>32767)
                  result_temp_ck_c2d_in5_w35 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w35 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w35 = result_final_temp_ck_c2d_in5_w35[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w36>32767 || result_final_temp_ck_c2d_in6_w36<-32768)begin
              if(result_final_temp_ck_c2d_in6_w36>32767)
                  result_temp_ck_c2d_in6_w36 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w36 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w36 = result_final_temp_ck_c2d_in6_w36[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w41>32767 || result_final_temp_ck_c2d_in1_w41<-32768)begin
              if(result_final_temp_ck_c2d_in1_w41>32767)
                  result_temp_ck_c2d_in1_w41 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w41 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w41 = result_final_temp_ck_c2d_in1_w41[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w42>32767 || result_final_temp_ck_c2d_in2_w42<-32768)begin
              if(result_final_temp_ck_c2d_in2_w42>32767)
                  result_temp_ck_c2d_in2_w42 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w42 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w42 = result_final_temp_ck_c2d_in2_w42[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w43>32767 || result_final_temp_ck_c2d_in3_w43<-32768)begin
              if(result_final_temp_ck_c2d_in3_w43>32767)
                  result_temp_ck_c2d_in3_w43 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w43 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w43 = result_final_temp_ck_c2d_in3_w43[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w44>32767 || result_final_temp_ck_c2d_in4_w44<-32768)begin
              if(result_final_temp_ck_c2d_in4_w44>32767)
                  result_temp_ck_c2d_in4_w44 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w44 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w44 = result_final_temp_ck_c2d_in4_w44[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w45>32767 || result_final_temp_ck_c2d_in5_w45<-32768)begin
              if(result_final_temp_ck_c2d_in5_w45>32767)
                  result_temp_ck_c2d_in5_w45 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w45 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w45 = result_final_temp_ck_c2d_in5_w45[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w46>32767 || result_final_temp_ck_c2d_in6_w46<-32768)begin
              if(result_final_temp_ck_c2d_in6_w46>32767)
                  result_temp_ck_c2d_in6_w46 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w46 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w46 = result_final_temp_ck_c2d_in6_w46[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w51>32767 || result_final_temp_ck_c2d_in1_w51<-32768)begin
              if(result_final_temp_ck_c2d_in1_w51>32767)
                  result_temp_ck_c2d_in1_w51 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w51 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w51 = result_final_temp_ck_c2d_in1_w51[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w52>32767 || result_final_temp_ck_c2d_in2_w52<-32768)begin
              if(result_final_temp_ck_c2d_in2_w52>32767)
                  result_temp_ck_c2d_in2_w52 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w52 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w52 = result_final_temp_ck_c2d_in2_w52[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w53>32767 || result_final_temp_ck_c2d_in3_w53<-32768)begin
              if(result_final_temp_ck_c2d_in3_w53>32767)
                  result_temp_ck_c2d_in3_w53 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w53 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w53 = result_final_temp_ck_c2d_in3_w53[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w54>32767 || result_final_temp_ck_c2d_in4_w54<-32768)begin
              if(result_final_temp_ck_c2d_in4_w54>32767)
                  result_temp_ck_c2d_in4_w54 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w54 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w54 = result_final_temp_ck_c2d_in4_w54[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w55>32767 || result_final_temp_ck_c2d_in5_w55<-32768)begin
              if(result_final_temp_ck_c2d_in5_w55>32767)
                  result_temp_ck_c2d_in5_w55 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w55 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w55 = result_final_temp_ck_c2d_in5_w55[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w56>32767 || result_final_temp_ck_c2d_in6_w56<-32768)begin
              if(result_final_temp_ck_c2d_in6_w56>32767)
                  result_temp_ck_c2d_in6_w56 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w56 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w56 = result_final_temp_ck_c2d_in6_w56[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w61>32767 || result_final_temp_ck_c2d_in1_w61<-32768)begin
              if(result_final_temp_ck_c2d_in1_w61>32767)
                  result_temp_ck_c2d_in1_w61 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w61 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w61 = result_final_temp_ck_c2d_in1_w61[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w62>32767 || result_final_temp_ck_c2d_in2_w62<-32768)begin
              if(result_final_temp_ck_c2d_in2_w62>32767)
                  result_temp_ck_c2d_in2_w62 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w62 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w62 = result_final_temp_ck_c2d_in2_w62[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w63>32767 || result_final_temp_ck_c2d_in3_w63<-32768)begin
              if(result_final_temp_ck_c2d_in3_w63>32767)
                  result_temp_ck_c2d_in3_w63 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w63 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w63 = result_final_temp_ck_c2d_in3_w63[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w64>32767 || result_final_temp_ck_c2d_in4_w64<-32768)begin
              if(result_final_temp_ck_c2d_in4_w64>32767)
                  result_temp_ck_c2d_in4_w64 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w64 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w64 = result_final_temp_ck_c2d_in4_w64[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w65>32767 || result_final_temp_ck_c2d_in5_w65<-32768)begin
              if(result_final_temp_ck_c2d_in5_w65>32767)
                  result_temp_ck_c2d_in5_w65 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w65 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w65 = result_final_temp_ck_c2d_in5_w65[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w66>32767 || result_final_temp_ck_c2d_in6_w66<-32768)begin
              if(result_final_temp_ck_c2d_in6_w66>32767)
                  result_temp_ck_c2d_in6_w66 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w66 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w66 = result_final_temp_ck_c2d_in6_w66[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w71>32767 || result_final_temp_ck_c2d_in1_w71<-32768)begin
              if(result_final_temp_ck_c2d_in1_w71>32767)
                  result_temp_ck_c2d_in1_w71 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w71 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w71 = result_final_temp_ck_c2d_in1_w71[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w72>32767 || result_final_temp_ck_c2d_in2_w72<-32768)begin
              if(result_final_temp_ck_c2d_in2_w72>32767)
                  result_temp_ck_c2d_in2_w72 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w72 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w72 = result_final_temp_ck_c2d_in2_w72[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w73>32767 || result_final_temp_ck_c2d_in3_w73<-32768)begin
              if(result_final_temp_ck_c2d_in3_w73>32767)
                  result_temp_ck_c2d_in3_w73 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w73 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w73 = result_final_temp_ck_c2d_in3_w73[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w74>32767 || result_final_temp_ck_c2d_in4_w74<-32768)begin
              if(result_final_temp_ck_c2d_in4_w74>32767)
                  result_temp_ck_c2d_in4_w74 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w74 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w74 = result_final_temp_ck_c2d_in4_w74[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w75>32767 || result_final_temp_ck_c2d_in5_w75<-32768)begin
              if(result_final_temp_ck_c2d_in5_w75>32767)
                  result_temp_ck_c2d_in5_w75 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w75 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w75 = result_final_temp_ck_c2d_in5_w75[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w76>32767 || result_final_temp_ck_c2d_in6_w76<-32768)begin
              if(result_final_temp_ck_c2d_in6_w76>32767)
                  result_temp_ck_c2d_in6_w76 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w76 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w76 = result_final_temp_ck_c2d_in6_w76[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w81>32767 || result_final_temp_ck_c2d_in1_w81<-32768)begin
              if(result_final_temp_ck_c2d_in1_w81>32767)
                  result_temp_ck_c2d_in1_w81 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w81 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w81 = result_final_temp_ck_c2d_in1_w81[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w82>32767 || result_final_temp_ck_c2d_in2_w82<-32768)begin
              if(result_final_temp_ck_c2d_in2_w82>32767)
                  result_temp_ck_c2d_in2_w82 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w82 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w82 = result_final_temp_ck_c2d_in2_w82[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w83>32767 || result_final_temp_ck_c2d_in3_w83<-32768)begin
              if(result_final_temp_ck_c2d_in3_w83>32767)
                  result_temp_ck_c2d_in3_w83 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w83 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w83 = result_final_temp_ck_c2d_in3_w83[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w84>32767 || result_final_temp_ck_c2d_in4_w84<-32768)begin
              if(result_final_temp_ck_c2d_in4_w84>32767)
                  result_temp_ck_c2d_in4_w84 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w84 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w84 = result_final_temp_ck_c2d_in4_w84[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w85>32767 || result_final_temp_ck_c2d_in5_w85<-32768)begin
              if(result_final_temp_ck_c2d_in5_w85>32767)
                  result_temp_ck_c2d_in5_w85 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w85 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w85 = result_final_temp_ck_c2d_in5_w85[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w86>32767 || result_final_temp_ck_c2d_in6_w86<-32768)begin
              if(result_final_temp_ck_c2d_in6_w86>32767)
                  result_temp_ck_c2d_in6_w86 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w86 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w86 = result_final_temp_ck_c2d_in6_w86[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w91>32767 || result_final_temp_ck_c2d_in1_w91<-32768)begin
              if(result_final_temp_ck_c2d_in1_w91>32767)
                  result_temp_ck_c2d_in1_w91 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w91 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w91 = result_final_temp_ck_c2d_in1_w91[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w92>32767 || result_final_temp_ck_c2d_in2_w92<-32768)begin
              if(result_final_temp_ck_c2d_in2_w92>32767)
                  result_temp_ck_c2d_in2_w92 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w92 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w92 = result_final_temp_ck_c2d_in2_w92[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w93>32767 || result_final_temp_ck_c2d_in3_w93<-32768)begin
              if(result_final_temp_ck_c2d_in3_w93>32767)
                  result_temp_ck_c2d_in3_w93 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w93 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w93 = result_final_temp_ck_c2d_in3_w93[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w94>32767 || result_final_temp_ck_c2d_in4_w94<-32768)begin
              if(result_final_temp_ck_c2d_in4_w94>32767)
                  result_temp_ck_c2d_in4_w94 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w94 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w94 = result_final_temp_ck_c2d_in4_w94[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w95>32767 || result_final_temp_ck_c2d_in5_w95<-32768)begin
              if(result_final_temp_ck_c2d_in5_w95>32767)
                  result_temp_ck_c2d_in5_w95 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w95 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w95 = result_final_temp_ck_c2d_in5_w95[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w96>32767 || result_final_temp_ck_c2d_in6_w96<-32768)begin
              if(result_final_temp_ck_c2d_in6_w96>32767)
                  result_temp_ck_c2d_in6_w96 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w96 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w96 = result_final_temp_ck_c2d_in6_w96[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w101>32767 || result_final_temp_ck_c2d_in1_w101<-32768)begin
              if(result_final_temp_ck_c2d_in1_w101>32767)
                  result_temp_ck_c2d_in1_w101 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w101 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w101 = result_final_temp_ck_c2d_in1_w101[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w102>32767 || result_final_temp_ck_c2d_in2_w102<-32768)begin
              if(result_final_temp_ck_c2d_in2_w102>32767)
                  result_temp_ck_c2d_in2_w102 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w102 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w102 = result_final_temp_ck_c2d_in2_w102[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w103>32767 || result_final_temp_ck_c2d_in3_w103<-32768)begin
              if(result_final_temp_ck_c2d_in3_w103>32767)
                  result_temp_ck_c2d_in3_w103 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w103 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w103 = result_final_temp_ck_c2d_in3_w103[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w104>32767 || result_final_temp_ck_c2d_in4_w104<-32768)begin
              if(result_final_temp_ck_c2d_in4_w104>32767)
                  result_temp_ck_c2d_in4_w104 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w104 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w104 = result_final_temp_ck_c2d_in4_w104[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w105>32767 || result_final_temp_ck_c2d_in5_w105<-32768)begin
              if(result_final_temp_ck_c2d_in5_w105>32767)
                  result_temp_ck_c2d_in5_w105 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w105 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w105 = result_final_temp_ck_c2d_in5_w105[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w106>32767 || result_final_temp_ck_c2d_in6_w106<-32768)begin
              if(result_final_temp_ck_c2d_in6_w106>32767)
                  result_temp_ck_c2d_in6_w106 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w106 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w106 = result_final_temp_ck_c2d_in6_w106[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w111>32767 || result_final_temp_ck_c2d_in1_w111<-32768)begin
              if(result_final_temp_ck_c2d_in1_w111>32767)
                  result_temp_ck_c2d_in1_w111 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w111 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w111 = result_final_temp_ck_c2d_in1_w111[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w112>32767 || result_final_temp_ck_c2d_in2_w112<-32768)begin
              if(result_final_temp_ck_c2d_in2_w112>32767)
                  result_temp_ck_c2d_in2_w112 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w112 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w112 = result_final_temp_ck_c2d_in2_w112[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w113>32767 || result_final_temp_ck_c2d_in3_w113<-32768)begin
              if(result_final_temp_ck_c2d_in3_w113>32767)
                  result_temp_ck_c2d_in3_w113 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w113 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w113 = result_final_temp_ck_c2d_in3_w113[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w114>32767 || result_final_temp_ck_c2d_in4_w114<-32768)begin
              if(result_final_temp_ck_c2d_in4_w114>32767)
                  result_temp_ck_c2d_in4_w114 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w114 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w114 = result_final_temp_ck_c2d_in4_w114[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w115>32767 || result_final_temp_ck_c2d_in5_w115<-32768)begin
              if(result_final_temp_ck_c2d_in5_w115>32767)
                  result_temp_ck_c2d_in5_w115 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w115 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w115 = result_final_temp_ck_c2d_in5_w115[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w116>32767 || result_final_temp_ck_c2d_in6_w116<-32768)begin
              if(result_final_temp_ck_c2d_in6_w116>32767)
                  result_temp_ck_c2d_in6_w116 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w116 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w116 = result_final_temp_ck_c2d_in6_w116[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w121>32767 || result_final_temp_ck_c2d_in1_w121<-32768)begin
              if(result_final_temp_ck_c2d_in1_w121>32767)
                  result_temp_ck_c2d_in1_w121 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w121 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w121 = result_final_temp_ck_c2d_in1_w121[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w122>32767 || result_final_temp_ck_c2d_in2_w122<-32768)begin
              if(result_final_temp_ck_c2d_in2_w122>32767)
                  result_temp_ck_c2d_in2_w122 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w122 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w122 = result_final_temp_ck_c2d_in2_w122[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w123>32767 || result_final_temp_ck_c2d_in3_w123<-32768)begin
              if(result_final_temp_ck_c2d_in3_w123>32767)
                  result_temp_ck_c2d_in3_w123 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w123 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w123 = result_final_temp_ck_c2d_in3_w123[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w124>32767 || result_final_temp_ck_c2d_in4_w124<-32768)begin
              if(result_final_temp_ck_c2d_in4_w124>32767)
                  result_temp_ck_c2d_in4_w124 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w124 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w124 = result_final_temp_ck_c2d_in4_w124[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w125>32767 || result_final_temp_ck_c2d_in5_w125<-32768)begin
              if(result_final_temp_ck_c2d_in5_w125>32767)
                  result_temp_ck_c2d_in5_w125 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w125 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w125 = result_final_temp_ck_c2d_in5_w125[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w126>32767 || result_final_temp_ck_c2d_in6_w126<-32768)begin
              if(result_final_temp_ck_c2d_in6_w126>32767)
                  result_temp_ck_c2d_in6_w126 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w126 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w126 = result_final_temp_ck_c2d_in6_w126[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w131>32767 || result_final_temp_ck_c2d_in1_w131<-32768)begin
              if(result_final_temp_ck_c2d_in1_w131>32767)
                  result_temp_ck_c2d_in1_w131 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w131 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w131 = result_final_temp_ck_c2d_in1_w131[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w132>32767 || result_final_temp_ck_c2d_in2_w132<-32768)begin
              if(result_final_temp_ck_c2d_in2_w132>32767)
                  result_temp_ck_c2d_in2_w132 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w132 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w132 = result_final_temp_ck_c2d_in2_w132[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w133>32767 || result_final_temp_ck_c2d_in3_w133<-32768)begin
              if(result_final_temp_ck_c2d_in3_w133>32767)
                  result_temp_ck_c2d_in3_w133 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w133 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w133 = result_final_temp_ck_c2d_in3_w133[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w134>32767 || result_final_temp_ck_c2d_in4_w134<-32768)begin
              if(result_final_temp_ck_c2d_in4_w134>32767)
                  result_temp_ck_c2d_in4_w134 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w134 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w134 = result_final_temp_ck_c2d_in4_w134[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w135>32767 || result_final_temp_ck_c2d_in5_w135<-32768)begin
              if(result_final_temp_ck_c2d_in5_w135>32767)
                  result_temp_ck_c2d_in5_w135 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w135 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w135 = result_final_temp_ck_c2d_in5_w135[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w136>32767 || result_final_temp_ck_c2d_in6_w136<-32768)begin
              if(result_final_temp_ck_c2d_in6_w136>32767)
                  result_temp_ck_c2d_in6_w136 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w136 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w136 = result_final_temp_ck_c2d_in6_w136[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w141>32767 || result_final_temp_ck_c2d_in1_w141<-32768)begin
              if(result_final_temp_ck_c2d_in1_w141>32767)
                  result_temp_ck_c2d_in1_w141 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w141 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w141 = result_final_temp_ck_c2d_in1_w141[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w142>32767 || result_final_temp_ck_c2d_in2_w142<-32768)begin
              if(result_final_temp_ck_c2d_in2_w142>32767)
                  result_temp_ck_c2d_in2_w142 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w142 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w142 = result_final_temp_ck_c2d_in2_w142[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w143>32767 || result_final_temp_ck_c2d_in3_w143<-32768)begin
              if(result_final_temp_ck_c2d_in3_w143>32767)
                  result_temp_ck_c2d_in3_w143 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w143 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w143 = result_final_temp_ck_c2d_in3_w143[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w144>32767 || result_final_temp_ck_c2d_in4_w144<-32768)begin
              if(result_final_temp_ck_c2d_in4_w144>32767)
                  result_temp_ck_c2d_in4_w144 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w144 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w144 = result_final_temp_ck_c2d_in4_w144[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w145>32767 || result_final_temp_ck_c2d_in5_w145<-32768)begin
              if(result_final_temp_ck_c2d_in5_w145>32767)
                  result_temp_ck_c2d_in5_w145 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w145 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w145 = result_final_temp_ck_c2d_in5_w145[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w146>32767 || result_final_temp_ck_c2d_in6_w146<-32768)begin
              if(result_final_temp_ck_c2d_in6_w146>32767)
                  result_temp_ck_c2d_in6_w146 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w146 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w146 = result_final_temp_ck_c2d_in6_w146[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w151>32767 || result_final_temp_ck_c2d_in1_w151<-32768)begin
              if(result_final_temp_ck_c2d_in1_w151>32767)
                  result_temp_ck_c2d_in1_w151 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w151 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w151 = result_final_temp_ck_c2d_in1_w151[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w152>32767 || result_final_temp_ck_c2d_in2_w152<-32768)begin
              if(result_final_temp_ck_c2d_in2_w152>32767)
                  result_temp_ck_c2d_in2_w152 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w152 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w152 = result_final_temp_ck_c2d_in2_w152[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w153>32767 || result_final_temp_ck_c2d_in3_w153<-32768)begin
              if(result_final_temp_ck_c2d_in3_w153>32767)
                  result_temp_ck_c2d_in3_w153 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w153 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w153 = result_final_temp_ck_c2d_in3_w153[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w154>32767 || result_final_temp_ck_c2d_in4_w154<-32768)begin
              if(result_final_temp_ck_c2d_in4_w154>32767)
                  result_temp_ck_c2d_in4_w154 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w154 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w154 = result_final_temp_ck_c2d_in4_w154[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w155>32767 || result_final_temp_ck_c2d_in5_w155<-32768)begin
              if(result_final_temp_ck_c2d_in5_w155>32767)
                  result_temp_ck_c2d_in5_w155 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w155 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w155 = result_final_temp_ck_c2d_in5_w155[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w156>32767 || result_final_temp_ck_c2d_in6_w156<-32768)begin
              if(result_final_temp_ck_c2d_in6_w156>32767)
                  result_temp_ck_c2d_in6_w156 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w156 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w156 = result_final_temp_ck_c2d_in6_w156[15:0];
          end
  
          if(result_final_temp_ck_c2d_in1_w161>32767 || result_final_temp_ck_c2d_in1_w161<-32768)begin
              if(result_final_temp_ck_c2d_in1_w161>32767)
                  result_temp_ck_c2d_in1_w161 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in1_w161 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in1_w161 = result_final_temp_ck_c2d_in1_w161[15:0];
          end
  
          if(result_final_temp_ck_c2d_in2_w162>32767 || result_final_temp_ck_c2d_in2_w162<-32768)begin
              if(result_final_temp_ck_c2d_in2_w162>32767)
                  result_temp_ck_c2d_in2_w162 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in2_w162 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in2_w162 = result_final_temp_ck_c2d_in2_w162[15:0];
          end
  
          if(result_final_temp_ck_c2d_in3_w163>32767 || result_final_temp_ck_c2d_in3_w163<-32768)begin
              if(result_final_temp_ck_c2d_in3_w163>32767)
                  result_temp_ck_c2d_in3_w163 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in3_w163 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in3_w163 = result_final_temp_ck_c2d_in3_w163[15:0];
          end
  
          if(result_final_temp_ck_c2d_in4_w164>32767 || result_final_temp_ck_c2d_in4_w164<-32768)begin
              if(result_final_temp_ck_c2d_in4_w164>32767)
                  result_temp_ck_c2d_in4_w164 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in4_w164 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in4_w164 = result_final_temp_ck_c2d_in4_w164[15:0];
          end
  
          if(result_final_temp_ck_c2d_in5_w165>32767 || result_final_temp_ck_c2d_in5_w165<-32768)begin
              if(result_final_temp_ck_c2d_in5_w165>32767)
                  result_temp_ck_c2d_in5_w165 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in5_w165 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in5_w165 = result_final_temp_ck_c2d_in5_w165[15:0];
          end
  
          if(result_final_temp_ck_c2d_in6_w166>32767 || result_final_temp_ck_c2d_in6_w166<-32768)begin
              if(result_final_temp_ck_c2d_in6_w166>32767)
                  result_temp_ck_c2d_in6_w166 = 16'b0111111111111111;
              else
                  result_temp_ck_c2d_in6_w166 = 16'b1000000000000000;
          end
          else begin
              result_temp_ck_c2d_in6_w166 = result_final_temp_ck_c2d_in6_w166[15:0];
          end
  
          done_ck_c2d=1'b1;
      end
      else begin
          result_temp_ck_c2d_in1_w11=0;
          result_temp_ck_c2d_in2_w12=0;
          result_temp_ck_c2d_in3_w13=0;
          result_temp_ck_c2d_in4_w14=0;
          result_temp_ck_c2d_in5_w15=0;
          result_temp_ck_c2d_in6_w16=0;
          result_temp_ck_c2d_in1_w21=0;
          result_temp_ck_c2d_in2_w22=0;
          result_temp_ck_c2d_in3_w23=0;
          result_temp_ck_c2d_in4_w24=0;
          result_temp_ck_c2d_in5_w25=0;
          result_temp_ck_c2d_in6_w26=0;
          result_temp_ck_c2d_in1_w31=0;
          result_temp_ck_c2d_in2_w32=0;
          result_temp_ck_c2d_in3_w33=0;
          result_temp_ck_c2d_in4_w34=0;
          result_temp_ck_c2d_in5_w35=0;
          result_temp_ck_c2d_in6_w36=0;
          result_temp_ck_c2d_in1_w41=0;
          result_temp_ck_c2d_in2_w42=0;
          result_temp_ck_c2d_in3_w43=0;
          result_temp_ck_c2d_in4_w44=0;
          result_temp_ck_c2d_in5_w45=0;
          result_temp_ck_c2d_in6_w46=0;
          result_temp_ck_c2d_in1_w51=0;
          result_temp_ck_c2d_in2_w52=0;
          result_temp_ck_c2d_in3_w53=0;
          result_temp_ck_c2d_in4_w54=0;
          result_temp_ck_c2d_in5_w55=0;
          result_temp_ck_c2d_in6_w56=0;
          result_temp_ck_c2d_in1_w61=0;
          result_temp_ck_c2d_in2_w62=0;
          result_temp_ck_c2d_in3_w63=0;
          result_temp_ck_c2d_in4_w64=0;
          result_temp_ck_c2d_in5_w65=0;
          result_temp_ck_c2d_in6_w66=0;
          result_temp_ck_c2d_in1_w71=0;
          result_temp_ck_c2d_in2_w72=0;
          result_temp_ck_c2d_in3_w73=0;
          result_temp_ck_c2d_in4_w74=0;
          result_temp_ck_c2d_in5_w75=0;
          result_temp_ck_c2d_in6_w76=0;
          result_temp_ck_c2d_in1_w81=0;
          result_temp_ck_c2d_in2_w82=0;
          result_temp_ck_c2d_in3_w83=0;
          result_temp_ck_c2d_in4_w84=0;
          result_temp_ck_c2d_in5_w85=0;
          result_temp_ck_c2d_in6_w86=0;
          result_temp_ck_c2d_in1_w91=0;
          result_temp_ck_c2d_in2_w92=0;
          result_temp_ck_c2d_in3_w93=0;
          result_temp_ck_c2d_in4_w94=0;
          result_temp_ck_c2d_in5_w95=0;
          result_temp_ck_c2d_in6_w96=0;
          result_temp_ck_c2d_in1_w101=0;
          result_temp_ck_c2d_in2_w102=0;
          result_temp_ck_c2d_in3_w103=0;
          result_temp_ck_c2d_in4_w104=0;
          result_temp_ck_c2d_in5_w105=0;
          result_temp_ck_c2d_in6_w106=0;
          result_temp_ck_c2d_in1_w111=0;
          result_temp_ck_c2d_in2_w112=0;
          result_temp_ck_c2d_in3_w113=0;
          result_temp_ck_c2d_in4_w114=0;
          result_temp_ck_c2d_in5_w115=0;
          result_temp_ck_c2d_in6_w116=0;
          result_temp_ck_c2d_in1_w121=0;
          result_temp_ck_c2d_in2_w122=0;
          result_temp_ck_c2d_in3_w123=0;
          result_temp_ck_c2d_in4_w124=0;
          result_temp_ck_c2d_in5_w125=0;
          result_temp_ck_c2d_in6_w126=0;
          result_temp_ck_c2d_in1_w131=0;
          result_temp_ck_c2d_in2_w132=0;
          result_temp_ck_c2d_in3_w133=0;
          result_temp_ck_c2d_in4_w134=0;
          result_temp_ck_c2d_in5_w135=0;
          result_temp_ck_c2d_in6_w136=0;
          result_temp_ck_c2d_in1_w141=0;
          result_temp_ck_c2d_in2_w142=0;
          result_temp_ck_c2d_in3_w143=0;
          result_temp_ck_c2d_in4_w144=0;
          result_temp_ck_c2d_in5_w145=0;
          result_temp_ck_c2d_in6_w146=0;
          result_temp_ck_c2d_in1_w151=0;
          result_temp_ck_c2d_in2_w152=0;
          result_temp_ck_c2d_in3_w153=0;
          result_temp_ck_c2d_in4_w154=0;
          result_temp_ck_c2d_in5_w155=0;
          result_temp_ck_c2d_in6_w156=0;
          result_temp_ck_c2d_in1_w161=0;
          result_temp_ck_c2d_in2_w162=0;
          result_temp_ck_c2d_in3_w163=0;
          result_temp_ck_c2d_in4_w164=0;
          result_temp_ck_c2d_in5_w165=0;
          result_temp_ck_c2d_in6_w166=0;
          done_ck_c2d=1'b0;
      end
  end
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w11(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w11),
  .Z_element(result_temp_m_c2d_in1_w11),
  .done(done_m_c2d_in1_w11)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w12(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w12),
  .Z_element(result_temp_m_c2d_in2_w12),
  .done(done_m_c2d_in2_w12)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w13(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w13),
  .Z_element(result_temp_m_c2d_in3_w13),
  .done(done_m_c2d_in3_w13)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w14(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w14),
  .Z_element(result_temp_m_c2d_in4_w14),
  .done(done_m_c2d_in4_w14)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w15(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w15),
  .Z_element(result_temp_m_c2d_in5_w15),
  .done(done_m_c2d_in5_w15)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w16(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w16),
  .Z_element(result_temp_m_c2d_in6_w16),
  .done(done_m_c2d_in6_w16)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w21(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w21),
  .Z_element(result_temp_m_c2d_in1_w21),
  .done(done_m_c2d_in1_w21)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w22(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w22),
  .Z_element(result_temp_m_c2d_in2_w22),
  .done(done_m_c2d_in2_w22)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w23(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w23),
  .Z_element(result_temp_m_c2d_in3_w23),
  .done(done_m_c2d_in3_w23)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w24(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w24),
  .Z_element(result_temp_m_c2d_in4_w24),
  .done(done_m_c2d_in4_w24)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w25(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w25),
  .Z_element(result_temp_m_c2d_in5_w25),
  .done(done_m_c2d_in5_w25)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w26(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w26),
  .Z_element(result_temp_m_c2d_in6_w26),
  .done(done_m_c2d_in6_w26)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w31(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w31),
  .Z_element(result_temp_m_c2d_in1_w31),
  .done(done_m_c2d_in1_w31)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w32(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w32),
  .Z_element(result_temp_m_c2d_in2_w32),
  .done(done_m_c2d_in2_w32)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w33(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w33),
  .Z_element(result_temp_m_c2d_in3_w33),
  .done(done_m_c2d_in3_w33)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w34(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w34),
  .Z_element(result_temp_m_c2d_in4_w34),
  .done(done_m_c2d_in4_w34)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w35(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w35),
  .Z_element(result_temp_m_c2d_in5_w35),
  .done(done_m_c2d_in5_w35)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w36(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w36),
  .Z_element(result_temp_m_c2d_in6_w36),
  .done(done_m_c2d_in6_w36)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w41(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w41),
  .Z_element(result_temp_m_c2d_in1_w41),
  .done(done_m_c2d_in1_w41)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w42(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w42),
  .Z_element(result_temp_m_c2d_in2_w42),
  .done(done_m_c2d_in2_w42)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w43(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w43),
  .Z_element(result_temp_m_c2d_in3_w43),
  .done(done_m_c2d_in3_w43)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w44(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w44),
  .Z_element(result_temp_m_c2d_in4_w44),
  .done(done_m_c2d_in4_w44)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w45(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w45),
  .Z_element(result_temp_m_c2d_in5_w45),
  .done(done_m_c2d_in5_w45)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w46(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w46),
  .Z_element(result_temp_m_c2d_in6_w46),
  .done(done_m_c2d_in6_w46)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w51(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w51),
  .Z_element(result_temp_m_c2d_in1_w51),
  .done(done_m_c2d_in1_w51)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w52(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w52),
  .Z_element(result_temp_m_c2d_in2_w52),
  .done(done_m_c2d_in2_w52)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w53(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w53),
  .Z_element(result_temp_m_c2d_in3_w53),
  .done(done_m_c2d_in3_w53)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w54(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w54),
  .Z_element(result_temp_m_c2d_in4_w54),
  .done(done_m_c2d_in4_w54)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w55(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w55),
  .Z_element(result_temp_m_c2d_in5_w55),
  .done(done_m_c2d_in5_w55)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w56(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w56),
  .Z_element(result_temp_m_c2d_in6_w56),
  .done(done_m_c2d_in6_w56)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w61(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w61),
  .Z_element(result_temp_m_c2d_in1_w61),
  .done(done_m_c2d_in1_w61)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w62(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w62),
  .Z_element(result_temp_m_c2d_in2_w62),
  .done(done_m_c2d_in2_w62)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w63(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w63),
  .Z_element(result_temp_m_c2d_in3_w63),
  .done(done_m_c2d_in3_w63)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w64(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w64),
  .Z_element(result_temp_m_c2d_in4_w64),
  .done(done_m_c2d_in4_w64)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w65(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w65),
  .Z_element(result_temp_m_c2d_in5_w65),
  .done(done_m_c2d_in5_w65)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w66(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w66),
  .Z_element(result_temp_m_c2d_in6_w66),
  .done(done_m_c2d_in6_w66)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w71(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w71),
  .Z_element(result_temp_m_c2d_in1_w71),
  .done(done_m_c2d_in1_w71)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w72(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w72),
  .Z_element(result_temp_m_c2d_in2_w72),
  .done(done_m_c2d_in2_w72)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w73(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w73),
  .Z_element(result_temp_m_c2d_in3_w73),
  .done(done_m_c2d_in3_w73)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w74(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w74),
  .Z_element(result_temp_m_c2d_in4_w74),
  .done(done_m_c2d_in4_w74)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w75(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w75),
  .Z_element(result_temp_m_c2d_in5_w75),
  .done(done_m_c2d_in5_w75)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w76(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w76),
  .Z_element(result_temp_m_c2d_in6_w76),
  .done(done_m_c2d_in6_w76)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w81(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w81),
  .Z_element(result_temp_m_c2d_in1_w81),
  .done(done_m_c2d_in1_w81)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w82(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w82),
  .Z_element(result_temp_m_c2d_in2_w82),
  .done(done_m_c2d_in2_w82)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w83(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w83),
  .Z_element(result_temp_m_c2d_in3_w83),
  .done(done_m_c2d_in3_w83)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w84(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w84),
  .Z_element(result_temp_m_c2d_in4_w84),
  .done(done_m_c2d_in4_w84)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w85(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w85),
  .Z_element(result_temp_m_c2d_in5_w85),
  .done(done_m_c2d_in5_w85)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w86(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w86),
  .Z_element(result_temp_m_c2d_in6_w86),
  .done(done_m_c2d_in6_w86)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w91(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w91),
  .Z_element(result_temp_m_c2d_in1_w91),
  .done(done_m_c2d_in1_w91)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w92(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w92),
  .Z_element(result_temp_m_c2d_in2_w92),
  .done(done_m_c2d_in2_w92)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w93(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w93),
  .Z_element(result_temp_m_c2d_in3_w93),
  .done(done_m_c2d_in3_w93)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w94(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w94),
  .Z_element(result_temp_m_c2d_in4_w94),
  .done(done_m_c2d_in4_w94)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w95(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w95),
  .Z_element(result_temp_m_c2d_in5_w95),
  .done(done_m_c2d_in5_w95)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w96(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w96),
  .Z_element(result_temp_m_c2d_in6_w96),
  .done(done_m_c2d_in6_w96)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w101(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w101),
  .Z_element(result_temp_m_c2d_in1_w101),
  .done(done_m_c2d_in1_w101)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w102(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w102),
  .Z_element(result_temp_m_c2d_in2_w102),
  .done(done_m_c2d_in2_w102)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w103(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w103),
  .Z_element(result_temp_m_c2d_in3_w103),
  .done(done_m_c2d_in3_w103)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w104(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w104),
  .Z_element(result_temp_m_c2d_in4_w104),
  .done(done_m_c2d_in4_w104)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w105(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w105),
  .Z_element(result_temp_m_c2d_in5_w105),
  .done(done_m_c2d_in5_w105)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w106(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w106),
  .Z_element(result_temp_m_c2d_in6_w106),
  .done(done_m_c2d_in6_w106)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w111(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w111),
  .Z_element(result_temp_m_c2d_in1_w111),
  .done(done_m_c2d_in1_w111)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w112(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w112),
  .Z_element(result_temp_m_c2d_in2_w112),
  .done(done_m_c2d_in2_w112)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w113(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w113),
  .Z_element(result_temp_m_c2d_in3_w113),
  .done(done_m_c2d_in3_w113)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w114(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w114),
  .Z_element(result_temp_m_c2d_in4_w114),
  .done(done_m_c2d_in4_w114)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w115(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w115),
  .Z_element(result_temp_m_c2d_in5_w115),
  .done(done_m_c2d_in5_w115)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w116(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w116),
  .Z_element(result_temp_m_c2d_in6_w116),
  .done(done_m_c2d_in6_w116)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w121(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w121),
  .Z_element(result_temp_m_c2d_in1_w121),
  .done(done_m_c2d_in1_w121)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w122(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w122),
  .Z_element(result_temp_m_c2d_in2_w122),
  .done(done_m_c2d_in2_w122)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w123(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w123),
  .Z_element(result_temp_m_c2d_in3_w123),
  .done(done_m_c2d_in3_w123)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w124(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w124),
  .Z_element(result_temp_m_c2d_in4_w124),
  .done(done_m_c2d_in4_w124)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w125(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w125),
  .Z_element(result_temp_m_c2d_in5_w125),
  .done(done_m_c2d_in5_w125)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w126(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w126),
  .Z_element(result_temp_m_c2d_in6_w126),
  .done(done_m_c2d_in6_w126)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w131(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w131),
  .Z_element(result_temp_m_c2d_in1_w131),
  .done(done_m_c2d_in1_w131)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w132(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w132),
  .Z_element(result_temp_m_c2d_in2_w132),
  .done(done_m_c2d_in2_w132)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w133(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w133),
  .Z_element(result_temp_m_c2d_in3_w133),
  .done(done_m_c2d_in3_w133)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w134(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w134),
  .Z_element(result_temp_m_c2d_in4_w134),
  .done(done_m_c2d_in4_w134)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w135(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w135),
  .Z_element(result_temp_m_c2d_in5_w135),
  .done(done_m_c2d_in5_w135)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w136(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w136),
  .Z_element(result_temp_m_c2d_in6_w136),
  .done(done_m_c2d_in6_w136)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w141(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w141),
  .Z_element(result_temp_m_c2d_in1_w141),
  .done(done_m_c2d_in1_w141)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w142(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w142),
  .Z_element(result_temp_m_c2d_in2_w142),
  .done(done_m_c2d_in2_w142)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w143(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w143),
  .Z_element(result_temp_m_c2d_in3_w143),
  .done(done_m_c2d_in3_w143)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w144(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w144),
  .Z_element(result_temp_m_c2d_in4_w144),
  .done(done_m_c2d_in4_w144)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w145(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w145),
  .Z_element(result_temp_m_c2d_in5_w145),
  .done(done_m_c2d_in5_w145)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w146(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w146),
  .Z_element(result_temp_m_c2d_in6_w146),
  .done(done_m_c2d_in6_w146)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w151(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w151),
  .Z_element(result_temp_m_c2d_in1_w151),
  .done(done_m_c2d_in1_w151)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w152(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w152),
  .Z_element(result_temp_m_c2d_in2_w152),
  .done(done_m_c2d_in2_w152)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w153(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w153),
  .Z_element(result_temp_m_c2d_in3_w153),
  .done(done_m_c2d_in3_w153)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w154(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w154),
  .Z_element(result_temp_m_c2d_in4_w154),
  .done(done_m_c2d_in4_w154)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w155(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w155),
  .Z_element(result_temp_m_c2d_in5_w155),
  .done(done_m_c2d_in5_w155)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w156(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w156),
  .Z_element(result_temp_m_c2d_in6_w156),
  .done(done_m_c2d_in6_w156)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in1_w161(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in1),
  .W_element(w_in_m_c2d_w161),
  .Z_element(result_temp_m_c2d_in1_w161),
  .done(done_m_c2d_in1_w161)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in2_w162(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in2),
  .W_element(w_in_m_c2d_w162),
  .Z_element(result_temp_m_c2d_in2_w162),
  .done(done_m_c2d_in2_w162)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in3_w163(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in3),
  .W_element(w_in_m_c2d_w163),
  .Z_element(result_temp_m_c2d_in3_w163),
  .done(done_m_c2d_in3_w163)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in4_w164(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in4),
  .W_element(w_in_m_c2d_w164),
  .Z_element(result_temp_m_c2d_in4_w164),
  .done(done_m_c2d_in4_w164)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in5_w165(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in5),
  .W_element(w_in_m_c2d_w165),
  .Z_element(result_temp_m_c2d_in5_w165),
  .done(done_m_c2d_in5_w165)
  );
  
  element_multiplier_c2#(BITWIDTH_M1,BITWIDTH_W,BITWIDTH_C2) multiply_c2_in6_w166(
  .clk(clk),
  .in_ready(en_m_c2d),
  .rst(rst_m_c2d),
  .X_element(x_in_m_c2d_ch_in6),
  .W_element(w_in_m_c2d_w166),
  .Z_element(result_temp_m_c2d_in6_w166),
  .done(done_m_c2d_in6_w166)
  );
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////Maxpool Layer 2///////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  assign start_addr_mk_m2=window_select_m2*2+(window_select_m2/5)*12;
  
  always@(posedge done_shift_m2 or posedge rst_m2)begin
      if(rst_m2)begin
          count_m2=0;
      end
      else begin
          if(count_m2<1)begin
              count_m2=count_m2+1;
          end
          else
              count_m2=2'hX;
      end
  end
  
  ////////////////////////////////// maxpool m2 //////////////////////////////////
  always@(*)begin
      case(state_m2)
          0:begin
              en_shift_m2=0;
              rst_shift_m2=1;
          end
          1:begin
              en_shift_m2=1;
              rst_shift_m2=0;
          end
          2:begin
              en_shift_m2=0;
              rst_shift_m2=1;
          end
          3:begin
              en_shift_m2=0;
              rst_shift_m2=0;
          end
          default:begin
              en_shift_m2=0;
              rst_shift_m2=1;
          end
      endcase
      if(count_m2==1)begin
          done_m2=1'b1;
      end
      else begin
          done_m2=1'b0;
      end
  end
  
      ////////////////////////////////// shift window m2 //////////////////////////////////
  always@(posedge done_mk_m2 or posedge rst_shift_m2)begin
      if(rst_shift_m2)begin
          count_shift_m2=0;
          din_buf_0_m2_ch1=0;
          din_buf_0_m2_ch2=0;
          din_buf_0_m2_ch3=0;
          din_buf_0_m2_ch4=0;
          din_buf_0_m2_ch5=0;
          din_buf_0_m2_ch6=0;
          din_buf_0_m2_ch7=0;
          din_buf_0_m2_ch8=0;
          din_buf_0_m2_ch9=0;
          din_buf_0_m2_ch10=0;
          din_buf_0_m2_ch11=0;
          din_buf_0_m2_ch12=0;
          din_buf_0_m2_ch13=0;
          din_buf_0_m2_ch14=0;
          din_buf_0_m2_ch15=0;
          din_buf_0_m2_ch16=0;
          a=1;
      end
      else begin
          if(count_shift_m2<25)begin
              if(count_shift_m2==a*5-1 && !done_mk_m2_row_ch1 && !done_mk_m2_row_ch2 && !done_mk_m2_row_ch3 && !done_mk_m2_row_ch4 && !done_mk_m2_row_ch5 && !done_mk_m2_row_ch6 && !done_mk_m2_row_ch7 && !done_mk_m2_row_ch8 && !done_mk_m2_row_ch9 && !done_mk_m2_row_ch10 && !done_mk_m2_row_ch11 && !done_mk_m2_row_ch12 && !done_mk_m2_row_ch13 && !done_mk_m2_row_ch14 && !done_mk_m2_row_ch15 && !done_mk_m2_row_ch16)begin
                  a=a+1;
                  done_mk_m2_row_ch1=1'b1;
                  done_mk_m2_row_ch2=1'b1;
                  done_mk_m2_row_ch3=1'b1;
                  done_mk_m2_row_ch4=1'b1;
                  done_mk_m2_row_ch5=1'b1;
                  done_mk_m2_row_ch6=1'b1;
                  done_mk_m2_row_ch7=1'b1;
                  done_mk_m2_row_ch8=1'b1;
                  done_mk_m2_row_ch9=1'b1;
                  done_mk_m2_row_ch10=1'b1;
                  done_mk_m2_row_ch11=1'b1;
                  done_mk_m2_row_ch12=1'b1;
                  done_mk_m2_row_ch13=1'b1;
                  done_mk_m2_row_ch14=1'b1;
                  done_mk_m2_row_ch15=1'b1;
                  done_mk_m2_row_ch16=1'b1;
              end
              else begin
                  count_shift_m2=count_shift_m2+1;
                  din_buf_0_m2_ch1=result_temp_mk_m2_ch1;
                  din_buf_0_m2_ch2=result_temp_mk_m2_ch2;
                  din_buf_0_m2_ch3=result_temp_mk_m2_ch3;
                  din_buf_0_m2_ch4=result_temp_mk_m2_ch4;
                  din_buf_0_m2_ch5=result_temp_mk_m2_ch5;
                  din_buf_0_m2_ch6=result_temp_mk_m2_ch6;
                  din_buf_0_m2_ch7=result_temp_mk_m2_ch7;
                  din_buf_0_m2_ch8=result_temp_mk_m2_ch8;
                  din_buf_0_m2_ch9=result_temp_mk_m2_ch9;
                  din_buf_0_m2_ch10=result_temp_mk_m2_ch10;
                  din_buf_0_m2_ch11=result_temp_mk_m2_ch11;
                  din_buf_0_m2_ch12=result_temp_mk_m2_ch12;
                  din_buf_0_m2_ch13=result_temp_mk_m2_ch13;
                  din_buf_0_m2_ch14=result_temp_mk_m2_ch14;
                  din_buf_0_m2_ch15=result_temp_mk_m2_ch15;
                  din_buf_0_m2_ch16=result_temp_mk_m2_ch16;
              end
          end
          else
              count_shift_m2=5'hX;
      end
  end
  
  always@(*)begin
      case(state_shift_m2)
          0:begin
              en_mk_m2=0;
              rst_mk_m2=1;
              window_select_m2=0;
          end
          1:begin
              en_mk_m2=1;
              rst_mk_m2=0;
              window_select_m2=count_shift_m2;
          end
          2:begin
              en_mk_m2=0;
              rst_mk_m2=1;
              window_select_m2=0;
          end
          default:begin
              en_mk_m2=0;
              rst_mk_m2=1;
              window_select_m2=0;
          end
      endcase
      if(count_shift_m2==25)begin
          done_shift_m2=1'b1;
      end
      else begin
          done_shift_m2=1'b0;
      end
  end
  
  always@(posedge rst_mk_m2)begin
      done_mk_m2_row_ch1=0;
      done_mk_m2_row_ch2=0;
      done_mk_m2_row_ch3=0;
      done_mk_m2_row_ch4=0;
      done_mk_m2_row_ch5=0;
      done_mk_m2_row_ch6=0;
      done_mk_m2_row_ch7=0;
      done_mk_m2_row_ch8=0;
      done_mk_m2_row_ch9=0;
      done_mk_m2_row_ch10=0;
      done_mk_m2_row_ch11=0;
      done_mk_m2_row_ch12=0;
      done_mk_m2_row_ch13=0;
      done_mk_m2_row_ch14=0;
      done_mk_m2_row_ch15=0;
      done_mk_m2_row_ch16=0;
  end
  
      /////////////////////////////////// maxppol kernel m2 ///////////////////////////////////
  
  assign element_mk_m2_ch1[0]=X_mk_m2_ch1[0:15];
  assign element_mk_m2_ch1[1]=X_mk_m2_ch1[16:31];
  assign element_mk_m2_ch1[2]=X_mk_m2_ch1[32:47];
  assign element_mk_m2_ch1[3]=X_mk_m2_ch1[48:63];
  assign element_mk_m2_ch1[4]=X_mk_m2_ch1[64:79];
  assign element_mk_m2_ch1[5]=X_mk_m2_ch1[80:95];
  assign element_mk_m2_ch1[6]=X_mk_m2_ch1[96:111];
  assign element_mk_m2_ch1[7]=X_mk_m2_ch1[112:127];
  assign element_mk_m2_ch1[8]=X_mk_m2_ch1[128:143];
  
  assign element_mk_m2_ch2[0]=X_mk_m2_ch2[0:15];
  assign element_mk_m2_ch2[1]=X_mk_m2_ch2[16:31];
  assign element_mk_m2_ch2[2]=X_mk_m2_ch2[32:47];
  assign element_mk_m2_ch2[3]=X_mk_m2_ch2[48:63];
  assign element_mk_m2_ch2[4]=X_mk_m2_ch2[64:79];
  assign element_mk_m2_ch2[5]=X_mk_m2_ch2[80:95];
  assign element_mk_m2_ch2[6]=X_mk_m2_ch2[96:111];
  assign element_mk_m2_ch2[7]=X_mk_m2_ch2[112:127];
  assign element_mk_m2_ch2[8]=X_mk_m2_ch2[128:143];
  
  assign element_mk_m2_ch3[0]=X_mk_m2_ch3[0:15];
  assign element_mk_m2_ch3[1]=X_mk_m2_ch3[16:31];
  assign element_mk_m2_ch3[2]=X_mk_m2_ch3[32:47];
  assign element_mk_m2_ch3[3]=X_mk_m2_ch3[48:63];
  assign element_mk_m2_ch3[4]=X_mk_m2_ch3[64:79];
  assign element_mk_m2_ch3[5]=X_mk_m2_ch3[80:95];
  assign element_mk_m2_ch3[6]=X_mk_m2_ch3[96:111];
  assign element_mk_m2_ch3[7]=X_mk_m2_ch3[112:127];
  assign element_mk_m2_ch3[8]=X_mk_m2_ch3[128:143];
  
  assign element_mk_m2_ch4[0]=X_mk_m2_ch4[0:15];
  assign element_mk_m2_ch4[1]=X_mk_m2_ch4[16:31];
  assign element_mk_m2_ch4[2]=X_mk_m2_ch4[32:47];
  assign element_mk_m2_ch4[3]=X_mk_m2_ch4[48:63];
  assign element_mk_m2_ch4[4]=X_mk_m2_ch4[64:79];
  assign element_mk_m2_ch4[5]=X_mk_m2_ch4[80:95];
  assign element_mk_m2_ch4[6]=X_mk_m2_ch4[96:111];
  assign element_mk_m2_ch4[7]=X_mk_m2_ch4[112:127];
  assign element_mk_m2_ch4[8]=X_mk_m2_ch4[128:143];
  
  assign element_mk_m2_ch5[0]=X_mk_m2_ch5[0:15];
  assign element_mk_m2_ch5[1]=X_mk_m2_ch5[16:31];
  assign element_mk_m2_ch5[2]=X_mk_m2_ch5[32:47];
  assign element_mk_m2_ch5[3]=X_mk_m2_ch5[48:63];
  assign element_mk_m2_ch5[4]=X_mk_m2_ch5[64:79];
  assign element_mk_m2_ch5[5]=X_mk_m2_ch5[80:95];
  assign element_mk_m2_ch5[6]=X_mk_m2_ch5[96:111];
  assign element_mk_m2_ch5[7]=X_mk_m2_ch5[112:127];
  assign element_mk_m2_ch5[8]=X_mk_m2_ch5[128:143];
  
  assign element_mk_m2_ch6[0]=X_mk_m2_ch6[0:15];
  assign element_mk_m2_ch6[1]=X_mk_m2_ch6[16:31];
  assign element_mk_m2_ch6[2]=X_mk_m2_ch6[32:47];
  assign element_mk_m2_ch6[3]=X_mk_m2_ch6[48:63];
  assign element_mk_m2_ch6[4]=X_mk_m2_ch6[64:79];
  assign element_mk_m2_ch6[5]=X_mk_m2_ch6[80:95];
  assign element_mk_m2_ch6[6]=X_mk_m2_ch6[96:111];
  assign element_mk_m2_ch6[7]=X_mk_m2_ch6[112:127];
  assign element_mk_m2_ch6[8]=X_mk_m2_ch6[128:143];
  
  assign element_mk_m2_ch7[0]=X_mk_m2_ch7[0:15];
  assign element_mk_m2_ch7[1]=X_mk_m2_ch7[16:31];
  assign element_mk_m2_ch7[2]=X_mk_m2_ch7[32:47];
  assign element_mk_m2_ch7[3]=X_mk_m2_ch7[48:63];
  assign element_mk_m2_ch7[4]=X_mk_m2_ch7[64:79];
  assign element_mk_m2_ch7[5]=X_mk_m2_ch7[80:95];
  assign element_mk_m2_ch7[6]=X_mk_m2_ch7[96:111];
  assign element_mk_m2_ch7[7]=X_mk_m2_ch7[112:127];
  assign element_mk_m2_ch7[8]=X_mk_m2_ch7[128:143];
  
  assign element_mk_m2_ch8[0]=X_mk_m2_ch8[0:15];
  assign element_mk_m2_ch8[1]=X_mk_m2_ch8[16:31];
  assign element_mk_m2_ch8[2]=X_mk_m2_ch8[32:47];
  assign element_mk_m2_ch8[3]=X_mk_m2_ch8[48:63];
  assign element_mk_m2_ch8[4]=X_mk_m2_ch8[64:79];
  assign element_mk_m2_ch8[5]=X_mk_m2_ch8[80:95];
  assign element_mk_m2_ch8[6]=X_mk_m2_ch8[96:111];
  assign element_mk_m2_ch8[7]=X_mk_m2_ch8[112:127];
  assign element_mk_m2_ch8[8]=X_mk_m2_ch8[128:143];
  
  assign element_mk_m2_ch9[0]=X_mk_m2_ch9[0:15];
  assign element_mk_m2_ch9[1]=X_mk_m2_ch9[16:31];
  assign element_mk_m2_ch9[2]=X_mk_m2_ch9[32:47];
  assign element_mk_m2_ch9[3]=X_mk_m2_ch9[48:63];
  assign element_mk_m2_ch9[4]=X_mk_m2_ch9[64:79];
  assign element_mk_m2_ch9[5]=X_mk_m2_ch9[80:95];
  assign element_mk_m2_ch9[6]=X_mk_m2_ch9[96:111];
  assign element_mk_m2_ch9[7]=X_mk_m2_ch9[112:127];
  assign element_mk_m2_ch9[8]=X_mk_m2_ch9[128:143];
  
  assign element_mk_m2_ch10[0]=X_mk_m2_ch10[0:15];
  assign element_mk_m2_ch10[1]=X_mk_m2_ch10[16:31];
  assign element_mk_m2_ch10[2]=X_mk_m2_ch10[32:47];
  assign element_mk_m2_ch10[3]=X_mk_m2_ch10[48:63];
  assign element_mk_m2_ch10[4]=X_mk_m2_ch10[64:79];
  assign element_mk_m2_ch10[5]=X_mk_m2_ch10[80:95];
  assign element_mk_m2_ch10[6]=X_mk_m2_ch10[96:111];
  assign element_mk_m2_ch10[7]=X_mk_m2_ch10[112:127];
  assign element_mk_m2_ch10[8]=X_mk_m2_ch10[128:143];
  
  assign element_mk_m2_ch11[0]=X_mk_m2_ch11[0:15];
  assign element_mk_m2_ch11[1]=X_mk_m2_ch11[16:31];
  assign element_mk_m2_ch11[2]=X_mk_m2_ch11[32:47];
  assign element_mk_m2_ch11[3]=X_mk_m2_ch11[48:63];
  assign element_mk_m2_ch11[4]=X_mk_m2_ch11[64:79];
  assign element_mk_m2_ch11[5]=X_mk_m2_ch11[80:95];
  assign element_mk_m2_ch11[6]=X_mk_m2_ch11[96:111];
  assign element_mk_m2_ch11[7]=X_mk_m2_ch11[112:127];
  assign element_mk_m2_ch11[8]=X_mk_m2_ch11[128:143];
  
  assign element_mk_m2_ch12[0]=X_mk_m2_ch12[0:15];
  assign element_mk_m2_ch12[1]=X_mk_m2_ch12[16:31];
  assign element_mk_m2_ch12[2]=X_mk_m2_ch12[32:47];
  assign element_mk_m2_ch12[3]=X_mk_m2_ch12[48:63];
  assign element_mk_m2_ch12[4]=X_mk_m2_ch12[64:79];
  assign element_mk_m2_ch12[5]=X_mk_m2_ch12[80:95];
  assign element_mk_m2_ch12[6]=X_mk_m2_ch12[96:111];
  assign element_mk_m2_ch12[7]=X_mk_m2_ch12[112:127];
  assign element_mk_m2_ch12[8]=X_mk_m2_ch12[128:143];
  
  assign element_mk_m2_ch13[0]=X_mk_m2_ch13[0:15];
  assign element_mk_m2_ch13[1]=X_mk_m2_ch13[16:31];
  assign element_mk_m2_ch13[2]=X_mk_m2_ch13[32:47];
  assign element_mk_m2_ch13[3]=X_mk_m2_ch13[48:63];
  assign element_mk_m2_ch13[4]=X_mk_m2_ch13[64:79];
  assign element_mk_m2_ch13[5]=X_mk_m2_ch13[80:95];
  assign element_mk_m2_ch13[6]=X_mk_m2_ch13[96:111];
  assign element_mk_m2_ch13[7]=X_mk_m2_ch13[112:127];
  assign element_mk_m2_ch13[8]=X_mk_m2_ch13[128:143];
  
  assign element_mk_m2_ch14[0]=X_mk_m2_ch14[0:15];
  assign element_mk_m2_ch14[1]=X_mk_m2_ch14[16:31];
  assign element_mk_m2_ch14[2]=X_mk_m2_ch14[32:47];
  assign element_mk_m2_ch14[3]=X_mk_m2_ch14[48:63];
  assign element_mk_m2_ch14[4]=X_mk_m2_ch14[64:79];
  assign element_mk_m2_ch14[5]=X_mk_m2_ch14[80:95];
  assign element_mk_m2_ch14[6]=X_mk_m2_ch14[96:111];
  assign element_mk_m2_ch14[7]=X_mk_m2_ch14[112:127];
  assign element_mk_m2_ch14[8]=X_mk_m2_ch14[128:143];
  
  assign element_mk_m2_ch15[0]=X_mk_m2_ch15[0:15];
  assign element_mk_m2_ch15[1]=X_mk_m2_ch15[16:31];
  assign element_mk_m2_ch15[2]=X_mk_m2_ch15[32:47];
  assign element_mk_m2_ch15[3]=X_mk_m2_ch15[48:63];
  assign element_mk_m2_ch15[4]=X_mk_m2_ch15[64:79];
  assign element_mk_m2_ch15[5]=X_mk_m2_ch15[80:95];
  assign element_mk_m2_ch15[6]=X_mk_m2_ch15[96:111];
  assign element_mk_m2_ch15[7]=X_mk_m2_ch15[112:127];
  assign element_mk_m2_ch15[8]=X_mk_m2_ch15[128:143];
  
  assign element_mk_m2_ch16[0]=X_mk_m2_ch16[0:15];
  assign element_mk_m2_ch16[1]=X_mk_m2_ch16[16:31];
  assign element_mk_m2_ch16[2]=X_mk_m2_ch16[32:47];
  assign element_mk_m2_ch16[3]=X_mk_m2_ch16[48:63];
  assign element_mk_m2_ch16[4]=X_mk_m2_ch16[64:79];
  assign element_mk_m2_ch16[5]=X_mk_m2_ch16[80:95];
  assign element_mk_m2_ch16[6]=X_mk_m2_ch16[96:111];
  assign element_mk_m2_ch16[7]=X_mk_m2_ch16[112:127];
  assign element_mk_m2_ch16[8]=X_mk_m2_ch16[128:143];
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////Dense Layer 1/////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  always@(posedge done_nc_d1 or posedge rst_d1)begin
      if(rst_d1)begin
          count_d1=0;
          buf0_ch1=0;
          buf1_ch1=0;
          buf2_ch1=0;
          buf3_ch1=0;
          buf4_ch1=0;
          buf5_ch1=0;
          buf6_ch1=0;
          buf7_ch1=0;
          buf8_ch1=0;
          buf9_ch1=0;
  
          buf0_ch2=0;
          buf1_ch2=0;
          buf2_ch2=0;
          buf3_ch2=0;
          buf4_ch2=0;
          buf5_ch2=0;
          buf6_ch2=0;
          buf7_ch2=0;
          buf8_ch2=0;
          buf9_ch2=0;
  
          buf0_ch3=0;
          buf1_ch3=0;
          buf2_ch3=0;
          buf3_ch3=0;
          buf4_ch3=0;
          buf5_ch3=0;
          buf6_ch3=0;
          buf7_ch3=0;
          buf8_ch3=0;
          buf9_ch3=0;
  
          buf0_ch4=0;
          buf1_ch4=0;
          buf2_ch4=0;
          buf3_ch4=0;
          buf4_ch4=0;
          buf5_ch4=0;
          buf6_ch4=0;
          buf7_ch4=0;
          buf8_ch4=0;
          buf9_ch4=0;
  
          buf0_ch5=0;
          buf1_ch5=0;
          buf2_ch5=0;
          buf3_ch5=0;
          buf4_ch5=0;
          buf5_ch5=0;
          buf6_ch5=0;
          buf7_ch5=0;
          buf8_ch5=0;
          buf9_ch5=0;
  
          buf0_ch6=0;
          buf1_ch6=0;
          buf2_ch6=0;
          buf3_ch6=0;
          buf4_ch6=0;
          buf5_ch6=0;
          buf6_ch6=0;
          buf7_ch6=0;
          buf8_ch6=0;
          buf9_ch6=0;
  
          buf0_ch7=0;
          buf1_ch7=0;
          buf2_ch7=0;
          buf3_ch7=0;
          buf4_ch7=0;
          buf5_ch7=0;
          buf6_ch7=0;
          buf7_ch7=0;
          buf8_ch7=0;
          buf9_ch7=0;
  
          buf0_ch8=0;
          buf1_ch8=0;
          buf2_ch8=0;
          buf3_ch8=0;
          buf4_ch8=0;
          buf5_ch8=0;
          buf6_ch8=0;
          buf7_ch8=0;
          buf8_ch8=0;
          buf9_ch8=0;
  
          buf0_ch9=0;
          buf1_ch9=0;
          buf2_ch9=0;
          buf3_ch9=0;
          buf4_ch9=0;
          buf5_ch9=0;
          buf6_ch9=0;
          buf7_ch9=0;
          buf8_ch9=0;
          buf9_ch9=0;
  
          buf0_ch10=0;
          buf1_ch10=0;
          buf2_ch10=0;
          buf3_ch10=0;
          buf4_ch10=0;
          buf5_ch10=0;
          buf6_ch10=0;
          buf7_ch10=0;
          buf8_ch10=0;
          buf9_ch10=0;
  
          buf0_ch11=0;
          buf1_ch11=0;
          buf2_ch11=0;
          buf3_ch11=0;
          buf4_ch11=0;
          buf5_ch11=0;
          buf6_ch11=0;
          buf7_ch11=0;
          buf8_ch11=0;
          buf9_ch11=0;
  
          buf0_ch12=0;
          buf1_ch12=0;
          buf2_ch12=0;
          buf3_ch12=0;
          buf4_ch12=0;
          buf5_ch12=0;
          buf6_ch12=0;
          buf7_ch12=0;
          buf8_ch12=0;
          buf9_ch12=0;
  
          buf0_ch13=0;
          buf1_ch13=0;
          buf2_ch13=0;
          buf3_ch13=0;
          buf4_ch13=0;
          buf5_ch13=0;
          buf6_ch13=0;
          buf7_ch13=0;
          buf8_ch13=0;
          buf9_ch13=0;
  
          buf0_ch14=0;
          buf1_ch14=0;
          buf2_ch14=0;
          buf3_ch14=0;
          buf4_ch14=0;
          buf5_ch14=0;
          buf6_ch14=0;
          buf7_ch14=0;
          buf8_ch14=0;
          buf9_ch14=0;
  
          buf0_ch15=0;
          buf1_ch15=0;
          buf2_ch15=0;
          buf3_ch15=0;
          buf4_ch15=0;
          buf5_ch15=0;
          buf6_ch15=0;
          buf7_ch15=0;
          buf8_ch15=0;
          buf9_ch15=0;
  
          buf0_ch16=0;
          buf1_ch16=0;
          buf2_ch16=0;
          buf3_ch16=0;
          buf4_ch16=0;
          buf5_ch16=0;
          buf6_ch16=0;
          buf7_ch16=0;
          buf8_ch16=0;
          buf9_ch16=0;
  
      end
      else begin
          if(count_d1<25)begin
              count_d1=count_d1+1;
              buf0_ch1=buf0_ch1+p0_ch1;
              buf1_ch1=buf1_ch1+p1_ch1;
              buf2_ch1=buf2_ch1+p2_ch1;
              buf3_ch1=buf3_ch1+p3_ch1;
              buf4_ch1=buf4_ch1+p4_ch1;
              buf5_ch1=buf5_ch1+p5_ch1;
              buf6_ch1=buf6_ch1+p6_ch1;
              buf7_ch1=buf7_ch1+p7_ch1;
              buf8_ch1=buf8_ch1+p8_ch1;
              buf9_ch1=buf9_ch1+p9_ch1;
  
              buf0_ch2=buf0_ch2+p0_ch2;
              buf1_ch2=buf1_ch2+p1_ch2;
              buf2_ch2=buf2_ch2+p2_ch2;
              buf3_ch2=buf3_ch2+p3_ch2;
              buf4_ch2=buf4_ch2+p4_ch2;
              buf5_ch2=buf5_ch2+p5_ch2;
              buf6_ch2=buf6_ch2+p6_ch2;
              buf7_ch2=buf7_ch2+p7_ch2;
              buf8_ch2=buf8_ch2+p8_ch2;
              buf9_ch2=buf9_ch2+p9_ch2;
  
              buf0_ch3=buf0_ch3+p0_ch3;
              buf1_ch3=buf1_ch3+p1_ch3;
              buf2_ch3=buf2_ch3+p2_ch3;
              buf3_ch3=buf3_ch3+p3_ch3;
              buf4_ch3=buf4_ch3+p4_ch3;
              buf5_ch3=buf5_ch3+p5_ch3;
              buf6_ch3=buf6_ch3+p6_ch3;
              buf7_ch3=buf7_ch3+p7_ch3;
              buf8_ch3=buf8_ch3+p8_ch3;
              buf9_ch3=buf9_ch3+p9_ch3;
  
              buf0_ch4=buf0_ch4+p0_ch4;
              buf1_ch4=buf1_ch4+p1_ch4;
              buf2_ch4=buf2_ch4+p2_ch4;
              buf3_ch4=buf3_ch4+p3_ch4;
              buf4_ch4=buf4_ch4+p4_ch4;
              buf5_ch4=buf5_ch4+p5_ch4;
              buf6_ch4=buf6_ch4+p6_ch4;
              buf7_ch4=buf7_ch4+p7_ch4;
              buf8_ch4=buf8_ch4+p8_ch4;
              buf9_ch4=buf9_ch4+p9_ch4;
  
              buf0_ch5=buf0_ch5+p0_ch5;
              buf1_ch5=buf1_ch5+p1_ch5;
              buf2_ch5=buf2_ch5+p2_ch5;
              buf3_ch5=buf3_ch5+p3_ch5;
              buf4_ch5=buf4_ch5+p4_ch5;
              buf5_ch5=buf5_ch5+p5_ch5;
              buf6_ch5=buf6_ch5+p6_ch5;
              buf7_ch5=buf7_ch5+p7_ch5;
              buf8_ch5=buf8_ch5+p8_ch5;
              buf9_ch5=buf9_ch5+p9_ch5;
  
              buf0_ch6=buf0_ch6+p0_ch6;
              buf1_ch6=buf1_ch6+p1_ch6;
              buf2_ch6=buf2_ch6+p2_ch6;
              buf3_ch6=buf3_ch6+p3_ch6;
              buf4_ch6=buf4_ch6+p4_ch6;
              buf5_ch6=buf5_ch6+p5_ch6;
              buf6_ch6=buf6_ch6+p6_ch6;
              buf7_ch6=buf7_ch6+p7_ch6;
              buf8_ch6=buf8_ch6+p8_ch6;
              buf9_ch6=buf9_ch6+p9_ch6;
  
              buf0_ch7=buf0_ch7+p0_ch7;
              buf1_ch7=buf1_ch7+p1_ch7;
              buf2_ch7=buf2_ch7+p2_ch7;
              buf3_ch7=buf3_ch7+p3_ch7;
              buf4_ch7=buf4_ch7+p4_ch7;
              buf5_ch7=buf5_ch7+p5_ch7;
              buf6_ch7=buf6_ch7+p6_ch7;
              buf7_ch7=buf7_ch7+p7_ch7;
              buf8_ch7=buf8_ch7+p8_ch7;
              buf9_ch7=buf9_ch7+p9_ch7;
  
              buf0_ch8=buf0_ch8+p0_ch8;
              buf1_ch8=buf1_ch8+p1_ch8;
              buf2_ch8=buf2_ch8+p2_ch8;
              buf3_ch8=buf3_ch8+p3_ch8;
              buf4_ch8=buf4_ch8+p4_ch8;
              buf5_ch8=buf5_ch8+p5_ch8;
              buf6_ch8=buf6_ch8+p6_ch8;
              buf7_ch8=buf7_ch8+p7_ch8;
              buf8_ch8=buf8_ch8+p8_ch8;
              buf9_ch8=buf9_ch8+p9_ch8;
  
              buf0_ch9=buf0_ch9+p0_ch9;
              buf1_ch9=buf1_ch9+p1_ch9;
              buf2_ch9=buf2_ch9+p2_ch9;
              buf3_ch9=buf3_ch9+p3_ch9;
              buf4_ch9=buf4_ch9+p4_ch9;
              buf5_ch9=buf5_ch9+p5_ch9;
              buf6_ch9=buf6_ch9+p6_ch9;
              buf7_ch9=buf7_ch9+p7_ch9;
              buf8_ch9=buf8_ch9+p8_ch9;
              buf9_ch9=buf9_ch9+p9_ch9;
  
              buf0_ch10=buf0_ch10+p0_ch10;
              buf1_ch10=buf1_ch10+p1_ch10;
              buf2_ch10=buf2_ch10+p2_ch10;
              buf3_ch10=buf3_ch10+p3_ch10;
              buf4_ch10=buf4_ch10+p4_ch10;
              buf5_ch10=buf5_ch10+p5_ch10;
              buf6_ch10=buf6_ch10+p6_ch10;
              buf7_ch10=buf7_ch10+p7_ch10;
              buf8_ch10=buf8_ch10+p8_ch10;
              buf9_ch10=buf9_ch10+p9_ch10;
  
              buf0_ch11=buf0_ch11+p0_ch11;
              buf1_ch11=buf1_ch11+p1_ch11;
              buf2_ch11=buf2_ch11+p2_ch11;
              buf3_ch11=buf3_ch11+p3_ch11;
              buf4_ch11=buf4_ch11+p4_ch11;
              buf5_ch11=buf5_ch11+p5_ch11;
              buf6_ch11=buf6_ch11+p6_ch11;
              buf7_ch11=buf7_ch11+p7_ch11;
              buf8_ch11=buf8_ch11+p8_ch11;
              buf9_ch11=buf9_ch11+p9_ch11;
  
              buf0_ch12=buf0_ch12+p0_ch12;
              buf1_ch12=buf1_ch12+p1_ch12;
              buf2_ch12=buf2_ch12+p2_ch12;
              buf3_ch12=buf3_ch12+p3_ch12;
              buf4_ch12=buf4_ch12+p4_ch12;
              buf5_ch12=buf5_ch12+p5_ch12;
              buf6_ch12=buf6_ch12+p6_ch12;
              buf7_ch12=buf7_ch12+p7_ch12;
              buf8_ch12=buf8_ch12+p8_ch12;
              buf9_ch12=buf9_ch12+p9_ch12;
  
              buf0_ch13=buf0_ch13+p0_ch13;
              buf1_ch13=buf1_ch13+p1_ch13;
              buf2_ch13=buf2_ch13+p2_ch13;
              buf3_ch13=buf3_ch13+p3_ch13;
              buf4_ch13=buf4_ch13+p4_ch13;
              buf5_ch13=buf5_ch13+p5_ch13;
              buf6_ch13=buf6_ch13+p6_ch13;
              buf7_ch13=buf7_ch13+p7_ch13;
              buf8_ch13=buf8_ch13+p8_ch13;
              buf9_ch13=buf9_ch13+p9_ch13;
  
              buf0_ch14=buf0_ch14+p0_ch14;
              buf1_ch14=buf1_ch14+p1_ch14;
              buf2_ch14=buf2_ch14+p2_ch14;
              buf3_ch14=buf3_ch14+p3_ch14;
              buf4_ch14=buf4_ch14+p4_ch14;
              buf5_ch14=buf5_ch14+p5_ch14;
              buf6_ch14=buf6_ch14+p6_ch14;
              buf7_ch14=buf7_ch14+p7_ch14;
              buf8_ch14=buf8_ch14+p8_ch14;
              buf9_ch14=buf9_ch14+p9_ch14;
  
              buf0_ch15=buf0_ch15+p0_ch15;
              buf1_ch15=buf1_ch15+p1_ch15;
              buf2_ch15=buf2_ch15+p2_ch15;
              buf3_ch15=buf3_ch15+p3_ch15;
              buf4_ch15=buf4_ch15+p4_ch15;
              buf5_ch15=buf5_ch15+p5_ch15;
              buf6_ch15=buf6_ch15+p6_ch15;
              buf7_ch15=buf7_ch15+p7_ch15;
              buf8_ch15=buf8_ch15+p8_ch15;
              buf9_ch15=buf9_ch15+p9_ch15;
  
              buf0_ch16=buf0_ch16+p0_ch16;
              buf1_ch16=buf1_ch16+p1_ch16;
              buf2_ch16=buf2_ch16+p2_ch16;
              buf3_ch16=buf3_ch16+p3_ch16;
              buf4_ch16=buf4_ch16+p4_ch16;
              buf5_ch16=buf5_ch16+p5_ch16;
              buf6_ch16=buf6_ch16+p6_ch16;
              buf7_ch16=buf7_ch16+p7_ch16;
              buf8_ch16=buf8_ch16+p8_ch16;
              buf9_ch16=buf9_ch16+p9_ch16;
  
          end
          else begin
              count_d1=5'hX;
              buf0_ch1=buf0_ch1;
              buf1_ch1=buf1_ch1;
              buf2_ch1=buf2_ch1;
              buf3_ch1=buf3_ch1;
              buf4_ch1=buf4_ch1;
              buf5_ch1=buf5_ch1;
              buf6_ch1=buf6_ch1;
              buf7_ch1=buf7_ch1;
              buf8_ch1=buf8_ch1;
              buf9_ch1=buf9_ch1;
  
              buf0_ch2=buf0_ch2;
              buf1_ch2=buf1_ch2;
              buf2_ch2=buf2_ch2;
              buf3_ch2=buf3_ch2;
              buf4_ch2=buf4_ch2;
              buf5_ch2=buf5_ch2;
              buf6_ch2=buf6_ch2;
              buf7_ch2=buf7_ch2;
              buf8_ch2=buf8_ch2;
              buf9_ch2=buf9_ch2;
  
              buf0_ch3=buf0_ch3;
              buf1_ch3=buf1_ch3;
              buf2_ch3=buf2_ch3;
              buf3_ch3=buf3_ch3;
              buf4_ch3=buf4_ch3;
              buf5_ch3=buf5_ch3;
              buf6_ch3=buf6_ch3;
              buf7_ch3=buf7_ch3;
              buf8_ch3=buf8_ch3;
              buf9_ch3=buf9_ch3;
  
              buf0_ch4=buf0_ch4;
              buf1_ch4=buf1_ch4;
              buf2_ch4=buf2_ch4;
              buf3_ch4=buf3_ch4;
              buf4_ch4=buf4_ch4;
              buf5_ch4=buf5_ch4;
              buf6_ch4=buf6_ch4;
              buf7_ch4=buf7_ch4;
              buf8_ch4=buf8_ch4;
              buf9_ch4=buf9_ch4;
  
              buf0_ch5=buf0_ch5;
              buf1_ch5=buf1_ch5;
              buf2_ch5=buf2_ch5;
              buf3_ch5=buf3_ch5;
              buf4_ch5=buf4_ch5;
              buf5_ch5=buf5_ch5;
              buf6_ch5=buf6_ch5;
              buf7_ch5=buf7_ch5;
              buf8_ch5=buf8_ch5;
              buf9_ch5=buf9_ch5;
  
              buf0_ch6=buf0_ch6;
              buf1_ch6=buf1_ch6;
              buf2_ch6=buf2_ch6;
              buf3_ch6=buf3_ch6;
              buf4_ch6=buf4_ch6;
              buf5_ch6=buf5_ch6;
              buf6_ch6=buf6_ch6;
              buf7_ch6=buf7_ch6;
              buf8_ch6=buf8_ch6;
              buf9_ch6=buf9_ch6;
  
              buf0_ch7=buf0_ch7;
              buf1_ch7=buf1_ch7;
              buf2_ch7=buf2_ch7;
              buf3_ch7=buf3_ch7;
              buf4_ch7=buf4_ch7;
              buf5_ch7=buf5_ch7;
              buf6_ch7=buf6_ch7;
              buf7_ch7=buf7_ch7;
              buf8_ch7=buf8_ch7;
              buf9_ch7=buf9_ch7;
  
              buf0_ch8=buf0_ch8;
              buf1_ch8=buf1_ch8;
              buf2_ch8=buf2_ch8;
              buf3_ch8=buf3_ch8;
              buf4_ch8=buf4_ch8;
              buf5_ch8=buf5_ch8;
              buf6_ch8=buf6_ch8;
              buf7_ch8=buf7_ch8;
              buf8_ch8=buf8_ch8;
              buf9_ch8=buf9_ch8;
  
              buf0_ch9=buf0_ch9;
              buf1_ch9=buf1_ch9;
              buf2_ch9=buf2_ch9;
              buf3_ch9=buf3_ch9;
              buf4_ch9=buf4_ch9;
              buf5_ch9=buf5_ch9;
              buf6_ch9=buf6_ch9;
              buf7_ch9=buf7_ch9;
              buf8_ch9=buf8_ch9;
              buf9_ch9=buf9_ch9;
  
              buf0_ch10=buf0_ch10;
              buf1_ch10=buf1_ch10;
              buf2_ch10=buf2_ch10;
              buf3_ch10=buf3_ch10;
              buf4_ch10=buf4_ch10;
              buf5_ch10=buf5_ch10;
              buf6_ch10=buf6_ch10;
              buf7_ch10=buf7_ch10;
              buf8_ch10=buf8_ch10;
              buf9_ch10=buf9_ch10;
  
              buf0_ch11=buf0_ch11;
              buf1_ch11=buf1_ch11;
              buf2_ch11=buf2_ch11;
              buf3_ch11=buf3_ch11;
              buf4_ch11=buf4_ch11;
              buf5_ch11=buf5_ch11;
              buf6_ch11=buf6_ch11;
              buf7_ch11=buf7_ch11;
              buf8_ch11=buf8_ch11;
              buf9_ch11=buf9_ch11;
  
              buf0_ch12=buf0_ch12;
              buf1_ch12=buf1_ch12;
              buf2_ch12=buf2_ch12;
              buf3_ch12=buf3_ch12;
              buf4_ch12=buf4_ch12;
              buf5_ch12=buf5_ch12;
              buf6_ch12=buf6_ch12;
              buf7_ch12=buf7_ch12;
              buf8_ch12=buf8_ch12;
              buf9_ch12=buf9_ch12;
  
              buf0_ch13=buf0_ch13;
              buf1_ch13=buf1_ch13;
              buf2_ch13=buf2_ch13;
              buf3_ch13=buf3_ch13;
              buf4_ch13=buf4_ch13;
              buf5_ch13=buf5_ch13;
              buf6_ch13=buf6_ch13;
              buf7_ch13=buf7_ch13;
              buf8_ch13=buf8_ch13;
              buf9_ch13=buf9_ch13;
  
              buf0_ch14=buf0_ch14;
              buf1_ch14=buf1_ch14;
              buf2_ch14=buf2_ch14;
              buf3_ch14=buf3_ch14;
              buf4_ch14=buf4_ch14;
              buf5_ch14=buf5_ch14;
              buf6_ch14=buf6_ch14;
              buf7_ch14=buf7_ch14;
              buf8_ch14=buf8_ch14;
              buf9_ch14=buf9_ch14;
  
              buf0_ch15=buf0_ch15;
              buf1_ch15=buf1_ch15;
              buf2_ch15=buf2_ch15;
              buf3_ch15=buf3_ch15;
              buf4_ch15=buf4_ch15;
              buf5_ch15=buf5_ch15;
              buf6_ch15=buf6_ch15;
              buf7_ch15=buf7_ch15;
              buf8_ch15=buf8_ch15;
              buf9_ch15=buf9_ch15;
  
              buf0_ch16=buf0_ch16;
              buf1_ch16=buf1_ch16;
              buf2_ch16=buf2_ch16;
              buf3_ch16=buf3_ch16;
              buf4_ch16=buf4_ch16;
              buf5_ch16=buf5_ch16;
              buf6_ch16=buf6_ch16;
              buf7_ch16=buf7_ch16;
              buf8_ch16=buf8_ch16;
              buf9_ch16=buf9_ch16;
  
          end
      end
  end
  
  always@(*)begin
      case(state_d1)
          0:begin
              en_nc_d1=0;
              rst_nc_d1=1;
              weight_select_d1=count_d1;
          end
          1:begin
              en_nc_d1=1;
              rst_nc_d1=0;
              weight_select_d1=count_d1;
          end
          2:begin
              en_nc_d1=0;
              rst_nc_d1=1;
              weight_select_d1=count_d1;
          end
          default:begin
              en_nc_d1=0;
              rst_nc_d1=1;
              weight_select_d1=count_d1;
          end
      endcase
      if(count_d1==25)begin
          fc_out[0:31]=buf0_ch1 + buf0_ch2 + buf0_ch3 + buf0_ch4 + buf0_ch5 + buf0_ch6 + buf0_ch7 + buf0_ch8 + buf0_ch9 + buf0_ch10 + buf0_ch11 + buf0_ch12 + buf0_ch13 + buf0_ch14 + buf0_ch15 + buf0_ch16 + (bias_d1[0:7]<<6);
          fc_out[32:63]=buf1_ch1 + buf1_ch2 + buf1_ch3 + buf1_ch4 + buf1_ch5 + buf1_ch6 + buf1_ch7 + buf1_ch8 + buf1_ch9 + buf1_ch10 + buf1_ch11 + buf1_ch12 + buf1_ch13 + buf1_ch14 + buf1_ch15 + buf1_ch16 + (bias_d1[8:15]<<6);
          fc_out[64:95]=buf2_ch1 + buf2_ch2 + buf2_ch3 + buf2_ch4 + buf2_ch5 + buf2_ch6 + buf2_ch7 + buf2_ch8 + buf2_ch9 + buf2_ch10 + buf2_ch11 + buf2_ch12 + buf2_ch13 + buf2_ch14 + buf2_ch15 + buf2_ch16 + (bias_d1[16:23]<<6);
          fc_out[96:127]=buf3_ch1 + buf3_ch2 + buf3_ch3 + buf3_ch4 + buf3_ch5 + buf3_ch6 + buf3_ch7 + buf3_ch8 + buf3_ch9 + buf3_ch10 + buf3_ch11 + buf3_ch12 + buf3_ch13 + buf3_ch14 + buf3_ch15 + buf3_ch16 + (bias_d1[24:31]<<6);
          fc_out[128:159]=buf4_ch1 + buf4_ch2 + buf4_ch3 + buf4_ch4 + buf4_ch5 + buf4_ch6 + buf4_ch7 + buf4_ch8 + buf4_ch9 + buf4_ch10 + buf4_ch11 + buf4_ch12 + buf4_ch13 + buf4_ch14 + buf4_ch15 + buf4_ch16 + (bias_d1[32:39]<<6);
          fc_out[160:191]=buf5_ch1 + buf5_ch2 + buf5_ch3 + buf5_ch4 + buf5_ch5 + buf5_ch6 + buf5_ch7 + buf5_ch8 + buf5_ch9 + buf5_ch10 + buf5_ch11 + buf5_ch12 + buf5_ch13 + buf5_ch14 + buf5_ch15 + buf5_ch16 + (bias_d1[40:47]<<6);
          fc_out[192:223]=buf6_ch1 + buf6_ch2 + buf6_ch3 + buf6_ch4 + buf6_ch5 + buf6_ch6 + buf6_ch7 + buf6_ch8 + buf6_ch9 + buf6_ch10 + buf6_ch11 + buf6_ch12 + buf6_ch13 + buf6_ch14 + buf6_ch15 + buf6_ch16 + (bias_d1[48:55]<<6);
          fc_out[224:255]=buf7_ch1 + buf7_ch2 + buf7_ch3 + buf7_ch4 + buf7_ch5 + buf7_ch6 + buf7_ch7 + buf7_ch8 + buf7_ch9 + buf7_ch10 + buf7_ch11 + buf7_ch12 + buf7_ch13 + buf7_ch14 + buf7_ch15 + buf7_ch16 + (bias_d1[56:63]<<6);
          fc_out[256:287]=buf8_ch1 + buf8_ch2 + buf8_ch3 + buf8_ch4 + buf8_ch5 + buf8_ch6 + buf8_ch7 + buf8_ch8 + buf8_ch9 + buf8_ch10 + buf8_ch11 + buf8_ch12 + buf8_ch13 + buf8_ch14 + buf8_ch15 + buf8_ch16 + (bias_d1[64:71]<<6);
          fc_out[288:319]=buf9_ch1 + buf9_ch2 + buf9_ch3 + buf9_ch4 + buf9_ch5 + buf9_ch6 + buf9_ch7 + buf9_ch8 + buf9_ch9 + buf9_ch10 + buf9_ch11 + buf9_ch12 + buf9_ch13 + buf9_ch14 + buf9_ch15 + buf9_ch16 + (bias_d1[72:79]<<6);
          done_den1=1'b1;
      end
      else begin
          fc_out=0;
          done_den1=1'b0;
      end
  end
  
      //////////////////////////////////// neuron calculation //////////////////////////////////
  
  assign mem_w_nc_d1_ch1[0]= w_neuron_nc_d1_ch1[0:7];
  assign mem_w_nc_d1_ch1[1]= w_neuron_nc_d1_ch1[8:15];
  assign mem_w_nc_d1_ch1[2]= w_neuron_nc_d1_ch1[16:23];
  assign mem_w_nc_d1_ch1[3]= w_neuron_nc_d1_ch1[24:31];
  assign mem_w_nc_d1_ch1[4]= w_neuron_nc_d1_ch1[32:39];
  assign mem_w_nc_d1_ch1[5]= w_neuron_nc_d1_ch1[40:47];
  assign mem_w_nc_d1_ch1[6]= w_neuron_nc_d1_ch1[48:55];
  assign mem_w_nc_d1_ch1[7]= w_neuron_nc_d1_ch1[56:63];
  assign mem_w_nc_d1_ch1[8]= w_neuron_nc_d1_ch1[64:71];
  assign mem_w_nc_d1_ch1[9]= w_neuron_nc_d1_ch1[72:79];
  
  assign mem_w_nc_d1_ch2[0]= w_neuron_nc_d1_ch2[0:7];
  assign mem_w_nc_d1_ch2[1]= w_neuron_nc_d1_ch2[8:15];
  assign mem_w_nc_d1_ch2[2]= w_neuron_nc_d1_ch2[16:23];
  assign mem_w_nc_d1_ch2[3]= w_neuron_nc_d1_ch2[24:31];
  assign mem_w_nc_d1_ch2[4]= w_neuron_nc_d1_ch2[32:39];
  assign mem_w_nc_d1_ch2[5]= w_neuron_nc_d1_ch2[40:47];
  assign mem_w_nc_d1_ch2[6]= w_neuron_nc_d1_ch2[48:55];
  assign mem_w_nc_d1_ch2[7]= w_neuron_nc_d1_ch2[56:63];
  assign mem_w_nc_d1_ch2[8]= w_neuron_nc_d1_ch2[64:71];
  assign mem_w_nc_d1_ch2[9]= w_neuron_nc_d1_ch2[72:79];
  
  assign mem_w_nc_d1_ch3[0]= w_neuron_nc_d1_ch3[0:7];
  assign mem_w_nc_d1_ch3[1]= w_neuron_nc_d1_ch3[8:15];
  assign mem_w_nc_d1_ch3[2]= w_neuron_nc_d1_ch3[16:23];
  assign mem_w_nc_d1_ch3[3]= w_neuron_nc_d1_ch3[24:31];
  assign mem_w_nc_d1_ch3[4]= w_neuron_nc_d1_ch3[32:39];
  assign mem_w_nc_d1_ch3[5]= w_neuron_nc_d1_ch3[40:47];
  assign mem_w_nc_d1_ch3[6]= w_neuron_nc_d1_ch3[48:55];
  assign mem_w_nc_d1_ch3[7]= w_neuron_nc_d1_ch3[56:63];
  assign mem_w_nc_d1_ch3[8]= w_neuron_nc_d1_ch3[64:71];
  assign mem_w_nc_d1_ch3[9]= w_neuron_nc_d1_ch3[72:79];
  
  assign mem_w_nc_d1_ch4[0]= w_neuron_nc_d1_ch4[0:7];
  assign mem_w_nc_d1_ch4[1]= w_neuron_nc_d1_ch4[8:15];
  assign mem_w_nc_d1_ch4[2]= w_neuron_nc_d1_ch4[16:23];
  assign mem_w_nc_d1_ch4[3]= w_neuron_nc_d1_ch4[24:31];
  assign mem_w_nc_d1_ch4[4]= w_neuron_nc_d1_ch4[32:39];
  assign mem_w_nc_d1_ch4[5]= w_neuron_nc_d1_ch4[40:47];
  assign mem_w_nc_d1_ch4[6]= w_neuron_nc_d1_ch4[48:55];
  assign mem_w_nc_d1_ch4[7]= w_neuron_nc_d1_ch4[56:63];
  assign mem_w_nc_d1_ch4[8]= w_neuron_nc_d1_ch4[64:71];
  assign mem_w_nc_d1_ch4[9]= w_neuron_nc_d1_ch4[72:79];
  
  assign mem_w_nc_d1_ch5[0]= w_neuron_nc_d1_ch5[0:7];
  assign mem_w_nc_d1_ch5[1]= w_neuron_nc_d1_ch5[8:15];
  assign mem_w_nc_d1_ch5[2]= w_neuron_nc_d1_ch5[16:23];
  assign mem_w_nc_d1_ch5[3]= w_neuron_nc_d1_ch5[24:31];
  assign mem_w_nc_d1_ch5[4]= w_neuron_nc_d1_ch5[32:39];
  assign mem_w_nc_d1_ch5[5]= w_neuron_nc_d1_ch5[40:47];
  assign mem_w_nc_d1_ch5[6]= w_neuron_nc_d1_ch5[48:55];
  assign mem_w_nc_d1_ch5[7]= w_neuron_nc_d1_ch5[56:63];
  assign mem_w_nc_d1_ch5[8]= w_neuron_nc_d1_ch5[64:71];
  assign mem_w_nc_d1_ch5[9]= w_neuron_nc_d1_ch5[72:79];
  
  assign mem_w_nc_d1_ch6[0]= w_neuron_nc_d1_ch6[0:7];
  assign mem_w_nc_d1_ch6[1]= w_neuron_nc_d1_ch6[8:15];
  assign mem_w_nc_d1_ch6[2]= w_neuron_nc_d1_ch6[16:23];
  assign mem_w_nc_d1_ch6[3]= w_neuron_nc_d1_ch6[24:31];
  assign mem_w_nc_d1_ch6[4]= w_neuron_nc_d1_ch6[32:39];
  assign mem_w_nc_d1_ch6[5]= w_neuron_nc_d1_ch6[40:47];
  assign mem_w_nc_d1_ch6[6]= w_neuron_nc_d1_ch6[48:55];
  assign mem_w_nc_d1_ch6[7]= w_neuron_nc_d1_ch6[56:63];
  assign mem_w_nc_d1_ch6[8]= w_neuron_nc_d1_ch6[64:71];
  assign mem_w_nc_d1_ch6[9]= w_neuron_nc_d1_ch6[72:79];
  
  assign mem_w_nc_d1_ch7[0]= w_neuron_nc_d1_ch7[0:7];
  assign mem_w_nc_d1_ch7[1]= w_neuron_nc_d1_ch7[8:15];
  assign mem_w_nc_d1_ch7[2]= w_neuron_nc_d1_ch7[16:23];
  assign mem_w_nc_d1_ch7[3]= w_neuron_nc_d1_ch7[24:31];
  assign mem_w_nc_d1_ch7[4]= w_neuron_nc_d1_ch7[32:39];
  assign mem_w_nc_d1_ch7[5]= w_neuron_nc_d1_ch7[40:47];
  assign mem_w_nc_d1_ch7[6]= w_neuron_nc_d1_ch7[48:55];
  assign mem_w_nc_d1_ch7[7]= w_neuron_nc_d1_ch7[56:63];
  assign mem_w_nc_d1_ch7[8]= w_neuron_nc_d1_ch7[64:71];
  assign mem_w_nc_d1_ch7[9]= w_neuron_nc_d1_ch7[72:79];
  
  assign mem_w_nc_d1_ch8[0]= w_neuron_nc_d1_ch8[0:7];
  assign mem_w_nc_d1_ch8[1]= w_neuron_nc_d1_ch8[8:15];
  assign mem_w_nc_d1_ch8[2]= w_neuron_nc_d1_ch8[16:23];
  assign mem_w_nc_d1_ch8[3]= w_neuron_nc_d1_ch8[24:31];
  assign mem_w_nc_d1_ch8[4]= w_neuron_nc_d1_ch8[32:39];
  assign mem_w_nc_d1_ch8[5]= w_neuron_nc_d1_ch8[40:47];
  assign mem_w_nc_d1_ch8[6]= w_neuron_nc_d1_ch8[48:55];
  assign mem_w_nc_d1_ch8[7]= w_neuron_nc_d1_ch8[56:63];
  assign mem_w_nc_d1_ch8[8]= w_neuron_nc_d1_ch8[64:71];
  assign mem_w_nc_d1_ch8[9]= w_neuron_nc_d1_ch8[72:79];
  
  assign mem_w_nc_d1_ch9[0]= w_neuron_nc_d1_ch9[0:7];
  assign mem_w_nc_d1_ch9[1]= w_neuron_nc_d1_ch9[8:15];
  assign mem_w_nc_d1_ch9[2]= w_neuron_nc_d1_ch9[16:23];
  assign mem_w_nc_d1_ch9[3]= w_neuron_nc_d1_ch9[24:31];
  assign mem_w_nc_d1_ch9[4]= w_neuron_nc_d1_ch9[32:39];
  assign mem_w_nc_d1_ch9[5]= w_neuron_nc_d1_ch9[40:47];
  assign mem_w_nc_d1_ch9[6]= w_neuron_nc_d1_ch9[48:55];
  assign mem_w_nc_d1_ch9[7]= w_neuron_nc_d1_ch9[56:63];
  assign mem_w_nc_d1_ch9[8]= w_neuron_nc_d1_ch9[64:71];
  assign mem_w_nc_d1_ch9[9]= w_neuron_nc_d1_ch9[72:79];
  
  assign mem_w_nc_d1_ch10[0]= w_neuron_nc_d1_ch10[0:7];
  assign mem_w_nc_d1_ch10[1]= w_neuron_nc_d1_ch10[8:15];
  assign mem_w_nc_d1_ch10[2]= w_neuron_nc_d1_ch10[16:23];
  assign mem_w_nc_d1_ch10[3]= w_neuron_nc_d1_ch10[24:31];
  assign mem_w_nc_d1_ch10[4]= w_neuron_nc_d1_ch10[32:39];
  assign mem_w_nc_d1_ch10[5]= w_neuron_nc_d1_ch10[40:47];
  assign mem_w_nc_d1_ch10[6]= w_neuron_nc_d1_ch10[48:55];
  assign mem_w_nc_d1_ch10[7]= w_neuron_nc_d1_ch10[56:63];
  assign mem_w_nc_d1_ch10[8]= w_neuron_nc_d1_ch10[64:71];
  assign mem_w_nc_d1_ch10[9]= w_neuron_nc_d1_ch10[72:79];
  
  assign mem_w_nc_d1_ch11[0]= w_neuron_nc_d1_ch11[0:7];
  assign mem_w_nc_d1_ch11[1]= w_neuron_nc_d1_ch11[8:15];
  assign mem_w_nc_d1_ch11[2]= w_neuron_nc_d1_ch11[16:23];
  assign mem_w_nc_d1_ch11[3]= w_neuron_nc_d1_ch11[24:31];
  assign mem_w_nc_d1_ch11[4]= w_neuron_nc_d1_ch11[32:39];
  assign mem_w_nc_d1_ch11[5]= w_neuron_nc_d1_ch11[40:47];
  assign mem_w_nc_d1_ch11[6]= w_neuron_nc_d1_ch11[48:55];
  assign mem_w_nc_d1_ch11[7]= w_neuron_nc_d1_ch11[56:63];
  assign mem_w_nc_d1_ch11[8]= w_neuron_nc_d1_ch11[64:71];
  assign mem_w_nc_d1_ch11[9]= w_neuron_nc_d1_ch11[72:79];
  
  assign mem_w_nc_d1_ch12[0]= w_neuron_nc_d1_ch12[0:7];
  assign mem_w_nc_d1_ch12[1]= w_neuron_nc_d1_ch12[8:15];
  assign mem_w_nc_d1_ch12[2]= w_neuron_nc_d1_ch12[16:23];
  assign mem_w_nc_d1_ch12[3]= w_neuron_nc_d1_ch12[24:31];
  assign mem_w_nc_d1_ch12[4]= w_neuron_nc_d1_ch12[32:39];
  assign mem_w_nc_d1_ch12[5]= w_neuron_nc_d1_ch12[40:47];
  assign mem_w_nc_d1_ch12[6]= w_neuron_nc_d1_ch12[48:55];
  assign mem_w_nc_d1_ch12[7]= w_neuron_nc_d1_ch12[56:63];
  assign mem_w_nc_d1_ch12[8]= w_neuron_nc_d1_ch12[64:71];
  assign mem_w_nc_d1_ch12[9]= w_neuron_nc_d1_ch12[72:79];
  
  assign mem_w_nc_d1_ch13[0]= w_neuron_nc_d1_ch13[0:7];
  assign mem_w_nc_d1_ch13[1]= w_neuron_nc_d1_ch13[8:15];
  assign mem_w_nc_d1_ch13[2]= w_neuron_nc_d1_ch13[16:23];
  assign mem_w_nc_d1_ch13[3]= w_neuron_nc_d1_ch13[24:31];
  assign mem_w_nc_d1_ch13[4]= w_neuron_nc_d1_ch13[32:39];
  assign mem_w_nc_d1_ch13[5]= w_neuron_nc_d1_ch13[40:47];
  assign mem_w_nc_d1_ch13[6]= w_neuron_nc_d1_ch13[48:55];
  assign mem_w_nc_d1_ch13[7]= w_neuron_nc_d1_ch13[56:63];
  assign mem_w_nc_d1_ch13[8]= w_neuron_nc_d1_ch13[64:71];
  assign mem_w_nc_d1_ch13[9]= w_neuron_nc_d1_ch13[72:79];
  
  assign mem_w_nc_d1_ch14[0]= w_neuron_nc_d1_ch14[0:7];
  assign mem_w_nc_d1_ch14[1]= w_neuron_nc_d1_ch14[8:15];
  assign mem_w_nc_d1_ch14[2]= w_neuron_nc_d1_ch14[16:23];
  assign mem_w_nc_d1_ch14[3]= w_neuron_nc_d1_ch14[24:31];
  assign mem_w_nc_d1_ch14[4]= w_neuron_nc_d1_ch14[32:39];
  assign mem_w_nc_d1_ch14[5]= w_neuron_nc_d1_ch14[40:47];
  assign mem_w_nc_d1_ch14[6]= w_neuron_nc_d1_ch14[48:55];
  assign mem_w_nc_d1_ch14[7]= w_neuron_nc_d1_ch14[56:63];
  assign mem_w_nc_d1_ch14[8]= w_neuron_nc_d1_ch14[64:71];
  assign mem_w_nc_d1_ch14[9]= w_neuron_nc_d1_ch14[72:79];
  
  assign mem_w_nc_d1_ch15[0]= w_neuron_nc_d1_ch15[0:7];
  assign mem_w_nc_d1_ch15[1]= w_neuron_nc_d1_ch15[8:15];
  assign mem_w_nc_d1_ch15[2]= w_neuron_nc_d1_ch15[16:23];
  assign mem_w_nc_d1_ch15[3]= w_neuron_nc_d1_ch15[24:31];
  assign mem_w_nc_d1_ch15[4]= w_neuron_nc_d1_ch15[32:39];
  assign mem_w_nc_d1_ch15[5]= w_neuron_nc_d1_ch15[40:47];
  assign mem_w_nc_d1_ch15[6]= w_neuron_nc_d1_ch15[48:55];
  assign mem_w_nc_d1_ch15[7]= w_neuron_nc_d1_ch15[56:63];
  assign mem_w_nc_d1_ch15[8]= w_neuron_nc_d1_ch15[64:71];
  assign mem_w_nc_d1_ch15[9]= w_neuron_nc_d1_ch15[72:79];
  
  assign mem_w_nc_d1_ch16[0]= w_neuron_nc_d1_ch16[0:7];
  assign mem_w_nc_d1_ch16[1]= w_neuron_nc_d1_ch16[8:15];
  assign mem_w_nc_d1_ch16[2]= w_neuron_nc_d1_ch16[16:23];
  assign mem_w_nc_d1_ch16[3]= w_neuron_nc_d1_ch16[24:31];
  assign mem_w_nc_d1_ch16[4]= w_neuron_nc_d1_ch16[32:39];
  assign mem_w_nc_d1_ch16[5]= w_neuron_nc_d1_ch16[40:47];
  assign mem_w_nc_d1_ch16[6]= w_neuron_nc_d1_ch16[48:55];
  assign mem_w_nc_d1_ch16[7]= w_neuron_nc_d1_ch16[56:63];
  assign mem_w_nc_d1_ch16[8]= w_neuron_nc_d1_ch16[64:71];
  assign mem_w_nc_d1_ch16[9]= w_neuron_nc_d1_ch16[72:79];
  
  always@(posedge (done_m_nc_d1_ch1 && done_m_nc_d1_ch2 && done_m_nc_d1_ch3 && done_m_nc_d1_ch4 && done_m_nc_d1_ch5 && done_m_nc_d1_ch6 && done_m_nc_d1_ch7 && done_m_nc_d1_ch8 && done_m_nc_d1_ch9 && done_m_nc_d1_ch10 && done_m_nc_d1_ch11 && done_m_nc_d1_ch12 && done_m_nc_d1_ch13 && done_m_nc_d1_ch14 && done_m_nc_d1_ch15 && done_m_nc_d1_ch16) or posedge rst_nc_d1)begin
      if(rst_nc_d1)begin
          count_nc_d1=0;
          buffer_nc_d1_ch1=0;
          buffer_nc_d1_ch2=0;
          buffer_nc_d1_ch3=0;
          buffer_nc_d1_ch4=0;
          buffer_nc_d1_ch5=0;
          buffer_nc_d1_ch6=0;
          buffer_nc_d1_ch7=0;
          buffer_nc_d1_ch8=0;
          buffer_nc_d1_ch9=0;
          buffer_nc_d1_ch10=0;
          buffer_nc_d1_ch11=0;
          buffer_nc_d1_ch12=0;
          buffer_nc_d1_ch13=0;
          buffer_nc_d1_ch14=0;
          buffer_nc_d1_ch15=0;
          buffer_nc_d1_ch16=0;
      end
      else begin
          if(count_nc_d1<10)begin
              count_nc_d1=count_nc_d1+1;
              buffer_nc_d1_ch1 = {buffer_nc_d1_ch1[32:319],result_temp_m_nc_d1_ch1};
              buffer_nc_d1_ch2 = {buffer_nc_d1_ch2[32:319],result_temp_m_nc_d1_ch2};
              buffer_nc_d1_ch3 = {buffer_nc_d1_ch3[32:319],result_temp_m_nc_d1_ch3};
              buffer_nc_d1_ch4 = {buffer_nc_d1_ch4[32:319],result_temp_m_nc_d1_ch4};
              buffer_nc_d1_ch5 = {buffer_nc_d1_ch5[32:319],result_temp_m_nc_d1_ch5};
              buffer_nc_d1_ch6 = {buffer_nc_d1_ch6[32:319],result_temp_m_nc_d1_ch6};
              buffer_nc_d1_ch7 = {buffer_nc_d1_ch7[32:319],result_temp_m_nc_d1_ch7};
              buffer_nc_d1_ch8 = {buffer_nc_d1_ch8[32:319],result_temp_m_nc_d1_ch8};
              buffer_nc_d1_ch9 = {buffer_nc_d1_ch9[32:319],result_temp_m_nc_d1_ch9};
              buffer_nc_d1_ch10 = {buffer_nc_d1_ch10[32:319],result_temp_m_nc_d1_ch10};
              buffer_nc_d1_ch11 = {buffer_nc_d1_ch11[32:319],result_temp_m_nc_d1_ch11};
              buffer_nc_d1_ch12 = {buffer_nc_d1_ch12[32:319],result_temp_m_nc_d1_ch12};
              buffer_nc_d1_ch13 = {buffer_nc_d1_ch13[32:319],result_temp_m_nc_d1_ch13};
              buffer_nc_d1_ch14 = {buffer_nc_d1_ch14[32:319],result_temp_m_nc_d1_ch14};
              buffer_nc_d1_ch15 = {buffer_nc_d1_ch15[32:319],result_temp_m_nc_d1_ch15};
              buffer_nc_d1_ch16 = {buffer_nc_d1_ch16[32:319],result_temp_m_nc_d1_ch16};
          end
      end
  end
  
  always@(*)begin
      case(state_nc_d1)
          0:begin
              en_m_nc_d1=0;
              rst_m_nc_d1=1;
              w_in_m_nc_d1_ch1=0;
              w_in_m_nc_d1_ch2=0;
              w_in_m_nc_d1_ch3=0;
              w_in_m_nc_d1_ch4=0;
              w_in_m_nc_d1_ch5=0;
              w_in_m_nc_d1_ch6=0;
              w_in_m_nc_d1_ch7=0;
              w_in_m_nc_d1_ch8=0;
              w_in_m_nc_d1_ch9=0;
              w_in_m_nc_d1_ch10=0;
              w_in_m_nc_d1_ch11=0;
              w_in_m_nc_d1_ch12=0;
              w_in_m_nc_d1_ch13=0;
              w_in_m_nc_d1_ch14=0;
              w_in_m_nc_d1_ch15=0;
              w_in_m_nc_d1_ch16=0;
          end
          1:begin
              en_m_nc_d1=1;
              rst_m_nc_d1=0;
              w_in_m_nc_d1_ch1=mem_w_nc_d1_ch1[count_nc_d1];
              w_in_m_nc_d1_ch2=mem_w_nc_d1_ch2[count_nc_d1];
              w_in_m_nc_d1_ch3=mem_w_nc_d1_ch3[count_nc_d1];
              w_in_m_nc_d1_ch4=mem_w_nc_d1_ch4[count_nc_d1];
              w_in_m_nc_d1_ch5=mem_w_nc_d1_ch5[count_nc_d1];
              w_in_m_nc_d1_ch6=mem_w_nc_d1_ch6[count_nc_d1];
              w_in_m_nc_d1_ch7=mem_w_nc_d1_ch7[count_nc_d1];
              w_in_m_nc_d1_ch8=mem_w_nc_d1_ch8[count_nc_d1];
              w_in_m_nc_d1_ch9=mem_w_nc_d1_ch9[count_nc_d1];
              w_in_m_nc_d1_ch10=mem_w_nc_d1_ch10[count_nc_d1];
              w_in_m_nc_d1_ch11=mem_w_nc_d1_ch11[count_nc_d1];
              w_in_m_nc_d1_ch12=mem_w_nc_d1_ch12[count_nc_d1];
              w_in_m_nc_d1_ch13=mem_w_nc_d1_ch13[count_nc_d1];
              w_in_m_nc_d1_ch14=mem_w_nc_d1_ch14[count_nc_d1];
              w_in_m_nc_d1_ch15=mem_w_nc_d1_ch15[count_nc_d1];
              w_in_m_nc_d1_ch16=mem_w_nc_d1_ch16[count_nc_d1];
          end
          2:begin
              en_m_nc_d1=0;
              rst_m_nc_d1=1;
              w_in_m_nc_d1_ch1=0;
              w_in_m_nc_d1_ch2=0;
              w_in_m_nc_d1_ch3=0;
              w_in_m_nc_d1_ch4=0;
              w_in_m_nc_d1_ch5=0;
              w_in_m_nc_d1_ch6=0;
              w_in_m_nc_d1_ch7=0;
              w_in_m_nc_d1_ch8=0;
              w_in_m_nc_d1_ch9=0;
              w_in_m_nc_d1_ch10=0;
              w_in_m_nc_d1_ch11=0;
              w_in_m_nc_d1_ch12=0;
              w_in_m_nc_d1_ch13=0;
              w_in_m_nc_d1_ch14=0;
              w_in_m_nc_d1_ch15=0;
              w_in_m_nc_d1_ch16=0;
          end
          default:begin
              en_m_nc_d1=0;
              rst_m_nc_d1=1;
              w_in_m_nc_d1_ch1=0;
              w_in_m_nc_d1_ch2=0;
              w_in_m_nc_d1_ch3=0;
              w_in_m_nc_d1_ch4=0;
              w_in_m_nc_d1_ch5=0;
              w_in_m_nc_d1_ch6=0;
              w_in_m_nc_d1_ch7=0;
              w_in_m_nc_d1_ch8=0;
              w_in_m_nc_d1_ch9=0;
              w_in_m_nc_d1_ch10=0;
              w_in_m_nc_d1_ch11=0;
              w_in_m_nc_d1_ch12=0;
              w_in_m_nc_d1_ch13=0;
              w_in_m_nc_d1_ch14=0;
              w_in_m_nc_d1_ch15=0;
              w_in_m_nc_d1_ch16=0;
          end
      endcase
      if (count_nc_d1==10)begin
          p0_ch1=buffer_nc_d1_ch1[0:31];
          p1_ch1=buffer_nc_d1_ch1[32:63];
          p2_ch1=buffer_nc_d1_ch1[64:95];
          p3_ch1=buffer_nc_d1_ch1[96:127];
          p4_ch1=buffer_nc_d1_ch1[128:159];
          p5_ch1=buffer_nc_d1_ch1[160:191];
          p6_ch1=buffer_nc_d1_ch1[192:223];
          p7_ch1=buffer_nc_d1_ch1[224:255];
          p8_ch1=buffer_nc_d1_ch1[256:287];
          p9_ch1=buffer_nc_d1_ch1[288:319];
  
          p0_ch2=buffer_nc_d1_ch2[0:31];
          p1_ch2=buffer_nc_d1_ch2[32:63];
          p2_ch2=buffer_nc_d1_ch2[64:95];
          p3_ch2=buffer_nc_d1_ch2[96:127];
          p4_ch2=buffer_nc_d1_ch2[128:159];
          p5_ch2=buffer_nc_d1_ch2[160:191];
          p6_ch2=buffer_nc_d1_ch2[192:223];
          p7_ch2=buffer_nc_d1_ch2[224:255];
          p8_ch2=buffer_nc_d1_ch2[256:287];
          p9_ch2=buffer_nc_d1_ch2[288:319];
  
          p0_ch3=buffer_nc_d1_ch3[0:31];
          p1_ch3=buffer_nc_d1_ch3[32:63];
          p2_ch3=buffer_nc_d1_ch3[64:95];
          p3_ch3=buffer_nc_d1_ch3[96:127];
          p4_ch3=buffer_nc_d1_ch3[128:159];
          p5_ch3=buffer_nc_d1_ch3[160:191];
          p6_ch3=buffer_nc_d1_ch3[192:223];
          p7_ch3=buffer_nc_d1_ch3[224:255];
          p8_ch3=buffer_nc_d1_ch3[256:287];
          p9_ch3=buffer_nc_d1_ch3[288:319];
  
          p0_ch4=buffer_nc_d1_ch4[0:31];
          p1_ch4=buffer_nc_d1_ch4[32:63];
          p2_ch4=buffer_nc_d1_ch4[64:95];
          p3_ch4=buffer_nc_d1_ch4[96:127];
          p4_ch4=buffer_nc_d1_ch4[128:159];
          p5_ch4=buffer_nc_d1_ch4[160:191];
          p6_ch4=buffer_nc_d1_ch4[192:223];
          p7_ch4=buffer_nc_d1_ch4[224:255];
          p8_ch4=buffer_nc_d1_ch4[256:287];
          p9_ch4=buffer_nc_d1_ch4[288:319];
  
          p0_ch5=buffer_nc_d1_ch5[0:31];
          p1_ch5=buffer_nc_d1_ch5[32:63];
          p2_ch5=buffer_nc_d1_ch5[64:95];
          p3_ch5=buffer_nc_d1_ch5[96:127];
          p4_ch5=buffer_nc_d1_ch5[128:159];
          p5_ch5=buffer_nc_d1_ch5[160:191];
          p6_ch5=buffer_nc_d1_ch5[192:223];
          p7_ch5=buffer_nc_d1_ch5[224:255];
          p8_ch5=buffer_nc_d1_ch5[256:287];
          p9_ch5=buffer_nc_d1_ch5[288:319];
  
          p0_ch6=buffer_nc_d1_ch6[0:31];
          p1_ch6=buffer_nc_d1_ch6[32:63];
          p2_ch6=buffer_nc_d1_ch6[64:95];
          p3_ch6=buffer_nc_d1_ch6[96:127];
          p4_ch6=buffer_nc_d1_ch6[128:159];
          p5_ch6=buffer_nc_d1_ch6[160:191];
          p6_ch6=buffer_nc_d1_ch6[192:223];
          p7_ch6=buffer_nc_d1_ch6[224:255];
          p8_ch6=buffer_nc_d1_ch6[256:287];
          p9_ch6=buffer_nc_d1_ch6[288:319];
  
          p0_ch7=buffer_nc_d1_ch7[0:31];
          p1_ch7=buffer_nc_d1_ch7[32:63];
          p2_ch7=buffer_nc_d1_ch7[64:95];
          p3_ch7=buffer_nc_d1_ch7[96:127];
          p4_ch7=buffer_nc_d1_ch7[128:159];
          p5_ch7=buffer_nc_d1_ch7[160:191];
          p6_ch7=buffer_nc_d1_ch7[192:223];
          p7_ch7=buffer_nc_d1_ch7[224:255];
          p8_ch7=buffer_nc_d1_ch7[256:287];
          p9_ch7=buffer_nc_d1_ch7[288:319];
  
          p0_ch8=buffer_nc_d1_ch8[0:31];
          p1_ch8=buffer_nc_d1_ch8[32:63];
          p2_ch8=buffer_nc_d1_ch8[64:95];
          p3_ch8=buffer_nc_d1_ch8[96:127];
          p4_ch8=buffer_nc_d1_ch8[128:159];
          p5_ch8=buffer_nc_d1_ch8[160:191];
          p6_ch8=buffer_nc_d1_ch8[192:223];
          p7_ch8=buffer_nc_d1_ch8[224:255];
          p8_ch8=buffer_nc_d1_ch8[256:287];
          p9_ch8=buffer_nc_d1_ch8[288:319];
  
          p0_ch9=buffer_nc_d1_ch9[0:31];
          p1_ch9=buffer_nc_d1_ch9[32:63];
          p2_ch9=buffer_nc_d1_ch9[64:95];
          p3_ch9=buffer_nc_d1_ch9[96:127];
          p4_ch9=buffer_nc_d1_ch9[128:159];
          p5_ch9=buffer_nc_d1_ch9[160:191];
          p6_ch9=buffer_nc_d1_ch9[192:223];
          p7_ch9=buffer_nc_d1_ch9[224:255];
          p8_ch9=buffer_nc_d1_ch9[256:287];
          p9_ch9=buffer_nc_d1_ch9[288:319];
  
          p0_ch10=buffer_nc_d1_ch10[0:31];
          p1_ch10=buffer_nc_d1_ch10[32:63];
          p2_ch10=buffer_nc_d1_ch10[64:95];
          p3_ch10=buffer_nc_d1_ch10[96:127];
          p4_ch10=buffer_nc_d1_ch10[128:159];
          p5_ch10=buffer_nc_d1_ch10[160:191];
          p6_ch10=buffer_nc_d1_ch10[192:223];
          p7_ch10=buffer_nc_d1_ch10[224:255];
          p8_ch10=buffer_nc_d1_ch10[256:287];
          p9_ch10=buffer_nc_d1_ch10[288:319];
  
          p0_ch11=buffer_nc_d1_ch11[0:31];
          p1_ch11=buffer_nc_d1_ch11[32:63];
          p2_ch11=buffer_nc_d1_ch11[64:95];
          p3_ch11=buffer_nc_d1_ch11[96:127];
          p4_ch11=buffer_nc_d1_ch11[128:159];
          p5_ch11=buffer_nc_d1_ch11[160:191];
          p6_ch11=buffer_nc_d1_ch11[192:223];
          p7_ch11=buffer_nc_d1_ch11[224:255];
          p8_ch11=buffer_nc_d1_ch11[256:287];
          p9_ch11=buffer_nc_d1_ch11[288:319];
  
          p0_ch12=buffer_nc_d1_ch12[0:31];
          p1_ch12=buffer_nc_d1_ch12[32:63];
          p2_ch12=buffer_nc_d1_ch12[64:95];
          p3_ch12=buffer_nc_d1_ch12[96:127];
          p4_ch12=buffer_nc_d1_ch12[128:159];
          p5_ch12=buffer_nc_d1_ch12[160:191];
          p6_ch12=buffer_nc_d1_ch12[192:223];
          p7_ch12=buffer_nc_d1_ch12[224:255];
          p8_ch12=buffer_nc_d1_ch12[256:287];
          p9_ch12=buffer_nc_d1_ch12[288:319];
  
          p0_ch13=buffer_nc_d1_ch13[0:31];
          p1_ch13=buffer_nc_d1_ch13[32:63];
          p2_ch13=buffer_nc_d1_ch13[64:95];
          p3_ch13=buffer_nc_d1_ch13[96:127];
          p4_ch13=buffer_nc_d1_ch13[128:159];
          p5_ch13=buffer_nc_d1_ch13[160:191];
          p6_ch13=buffer_nc_d1_ch13[192:223];
          p7_ch13=buffer_nc_d1_ch13[224:255];
          p8_ch13=buffer_nc_d1_ch13[256:287];
          p9_ch13=buffer_nc_d1_ch13[288:319];
  
          p0_ch14=buffer_nc_d1_ch14[0:31];
          p1_ch14=buffer_nc_d1_ch14[32:63];
          p2_ch14=buffer_nc_d1_ch14[64:95];
          p3_ch14=buffer_nc_d1_ch14[96:127];
          p4_ch14=buffer_nc_d1_ch14[128:159];
          p5_ch14=buffer_nc_d1_ch14[160:191];
          p6_ch14=buffer_nc_d1_ch14[192:223];
          p7_ch14=buffer_nc_d1_ch14[224:255];
          p8_ch14=buffer_nc_d1_ch14[256:287];
          p9_ch14=buffer_nc_d1_ch14[288:319];
  
          p0_ch15=buffer_nc_d1_ch15[0:31];
          p1_ch15=buffer_nc_d1_ch15[32:63];
          p2_ch15=buffer_nc_d1_ch15[64:95];
          p3_ch15=buffer_nc_d1_ch15[96:127];
          p4_ch15=buffer_nc_d1_ch15[128:159];
          p5_ch15=buffer_nc_d1_ch15[160:191];
          p6_ch15=buffer_nc_d1_ch15[192:223];
          p7_ch15=buffer_nc_d1_ch15[224:255];
          p8_ch15=buffer_nc_d1_ch15[256:287];
          p9_ch15=buffer_nc_d1_ch15[288:319];
  
          p0_ch16=buffer_nc_d1_ch16[0:31];
          p1_ch16=buffer_nc_d1_ch16[32:63];
          p2_ch16=buffer_nc_d1_ch16[64:95];
          p3_ch16=buffer_nc_d1_ch16[96:127];
          p4_ch16=buffer_nc_d1_ch16[128:159];
          p5_ch16=buffer_nc_d1_ch16[160:191];
          p6_ch16=buffer_nc_d1_ch16[192:223];
          p7_ch16=buffer_nc_d1_ch16[224:255];
          p8_ch16=buffer_nc_d1_ch16[256:287];
          p9_ch16=buffer_nc_d1_ch16[288:319];
  
          done_nc_d1=1'b1;
      end
      else begin
          p0_ch1=0;
          p1_ch1=0;
          p2_ch1=0;
          p3_ch1=0;
          p4_ch1=0;
          p5_ch1=0;
          p6_ch1=0;
          p7_ch1=0;
          p8_ch1=0;
          p9_ch1=0;
  
          p0_ch2=0;
          p1_ch2=0;
          p2_ch2=0;
          p3_ch2=0;
          p4_ch2=0;
          p5_ch2=0;
          p6_ch2=0;
          p7_ch2=0;
          p8_ch2=0;
          p9_ch2=0;
  
          p0_ch3=0;
          p1_ch3=0;
          p2_ch3=0;
          p3_ch3=0;
          p4_ch3=0;
          p5_ch3=0;
          p6_ch3=0;
          p7_ch3=0;
          p8_ch3=0;
          p9_ch3=0;
  
          p0_ch4=0;
          p1_ch4=0;
          p2_ch4=0;
          p3_ch4=0;
          p4_ch4=0;
          p5_ch4=0;
          p6_ch4=0;
          p7_ch4=0;
          p8_ch4=0;
          p9_ch4=0;
  
          p0_ch5=0;
          p1_ch5=0;
          p2_ch5=0;
          p3_ch5=0;
          p4_ch5=0;
          p5_ch5=0;
          p6_ch5=0;
          p7_ch5=0;
          p8_ch5=0;
          p9_ch5=0;
  
          p0_ch6=0;
          p1_ch6=0;
          p2_ch6=0;
          p3_ch6=0;
          p4_ch6=0;
          p5_ch6=0;
          p6_ch6=0;
          p7_ch6=0;
          p8_ch6=0;
          p9_ch6=0;
  
          p0_ch7=0;
          p1_ch7=0;
          p2_ch7=0;
          p3_ch7=0;
          p4_ch7=0;
          p5_ch7=0;
          p6_ch7=0;
          p7_ch7=0;
          p8_ch7=0;
          p9_ch7=0;
  
          p0_ch8=0;
          p1_ch8=0;
          p2_ch8=0;
          p3_ch8=0;
          p4_ch8=0;
          p5_ch8=0;
          p6_ch8=0;
          p7_ch8=0;
          p8_ch8=0;
          p9_ch8=0;
  
          p0_ch9=0;
          p1_ch9=0;
          p2_ch9=0;
          p3_ch9=0;
          p4_ch9=0;
          p5_ch9=0;
          p6_ch9=0;
          p7_ch9=0;
          p8_ch9=0;
          p9_ch9=0;
  
          p0_ch10=0;
          p1_ch10=0;
          p2_ch10=0;
          p3_ch10=0;
          p4_ch10=0;
          p5_ch10=0;
          p6_ch10=0;
          p7_ch10=0;
          p8_ch10=0;
          p9_ch10=0;
  
          p0_ch11=0;
          p1_ch11=0;
          p2_ch11=0;
          p3_ch11=0;
          p4_ch11=0;
          p5_ch11=0;
          p6_ch11=0;
          p7_ch11=0;
          p8_ch11=0;
          p9_ch11=0;
  
          p0_ch12=0;
          p1_ch12=0;
          p2_ch12=0;
          p3_ch12=0;
          p4_ch12=0;
          p5_ch12=0;
          p6_ch12=0;
          p7_ch12=0;
          p8_ch12=0;
          p9_ch12=0;
  
          p0_ch13=0;
          p1_ch13=0;
          p2_ch13=0;
          p3_ch13=0;
          p4_ch13=0;
          p5_ch13=0;
          p6_ch13=0;
          p7_ch13=0;
          p8_ch13=0;
          p9_ch13=0;
  
          p0_ch14=0;
          p1_ch14=0;
          p2_ch14=0;
          p3_ch14=0;
          p4_ch14=0;
          p5_ch14=0;
          p6_ch14=0;
          p7_ch14=0;
          p8_ch14=0;
          p9_ch14=0;
  
          p0_ch15=0;
          p1_ch15=0;
          p2_ch15=0;
          p3_ch15=0;
          p4_ch15=0;
          p5_ch15=0;
          p6_ch15=0;
          p7_ch15=0;
          p8_ch15=0;
          p9_ch15=0;
  
          p0_ch16=0;
          p1_ch16=0;
          p2_ch16=0;
          p3_ch16=0;
          p4_ch16=0;
          p5_ch16=0;
          p6_ch16=0;
          p7_ch16=0;
          p8_ch16=0;
          p9_ch16=0;
  
          done_nc_d1=1'b0;
      end
  end
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch1(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch1),
  .W_element(w_in_m_nc_d1_ch1),
  .Z_element(result_temp_m_nc_d1_ch1),
  .done(done_m_nc_d1_ch1)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch2(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch2),
  .W_element(w_in_m_nc_d1_ch2),
  .Z_element(result_temp_m_nc_d1_ch2),
  .done(done_m_nc_d1_ch2)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch3(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch3),
  .W_element(w_in_m_nc_d1_ch3),
  .Z_element(result_temp_m_nc_d1_ch3),
  .done(done_m_nc_d1_ch3)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch4(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch4),
  .W_element(w_in_m_nc_d1_ch4),
  .Z_element(result_temp_m_nc_d1_ch4),
  .done(done_m_nc_d1_ch4)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch5(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch5),
  .W_element(w_in_m_nc_d1_ch5),
  .Z_element(result_temp_m_nc_d1_ch5),
  .done(done_m_nc_d1_ch5)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch6(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch6),
  .W_element(w_in_m_nc_d1_ch6),
  .Z_element(result_temp_m_nc_d1_ch6),
  .done(done_m_nc_d1_ch6)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch7(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch7),
  .W_element(w_in_m_nc_d1_ch7),
  .Z_element(result_temp_m_nc_d1_ch7),
  .done(done_m_nc_d1_ch7)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch8(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch8),
  .W_element(w_in_m_nc_d1_ch8),
  .Z_element(result_temp_m_nc_d1_ch8),
  .done(done_m_nc_d1_ch8)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch9(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch9),
  .W_element(w_in_m_nc_d1_ch9),
  .Z_element(result_temp_m_nc_d1_ch9),
  .done(done_m_nc_d1_ch9)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch10(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch10),
  .W_element(w_in_m_nc_d1_ch10),
  .Z_element(result_temp_m_nc_d1_ch10),
  .done(done_m_nc_d1_ch10)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch11(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch11),
  .W_element(w_in_m_nc_d1_ch11),
  .Z_element(result_temp_m_nc_d1_ch11),
  .done(done_m_nc_d1_ch11)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch12(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch12),
  .W_element(w_in_m_nc_d1_ch12),
  .Z_element(result_temp_m_nc_d1_ch12),
  .done(done_m_nc_d1_ch12)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch13(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch13),
  .W_element(w_in_m_nc_d1_ch13),
  .Z_element(result_temp_m_nc_d1_ch13),
  .done(done_m_nc_d1_ch13)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch14(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch14),
  .W_element(w_in_m_nc_d1_ch14),
  .Z_element(result_temp_m_nc_d1_ch14),
  .done(done_m_nc_d1_ch14)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch15(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch15),
  .W_element(w_in_m_nc_d1_ch15),
  .Z_element(result_temp_m_nc_d1_ch15),
  .done(done_m_nc_d1_ch15)
  );
  
  element_multiplier_d1#(BITWIDTH_M2,BITWIDTH_W,BITWIDTH_D1) element_mult_d1_ch16(
  .clk(clk),
  .in_ready(en_m_nc_d1),
  .rst(rst_m_nc_d1),
  .X_element(X_nc_d1_ch16),
  .W_element(w_in_m_nc_d1_ch16),
  .Z_element(result_temp_m_nc_d1_ch16),
  .done(done_m_nc_d1_ch16)
  );
  
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////Softmax Layer/////////////////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  always@(max)begin
      if(i==10)begin
          prediction=temp;
          done_soft=1'b1;
      end
      else begin
          prediction=4'hX;
          done_soft=1'b0;
      end
  end
endmodule
