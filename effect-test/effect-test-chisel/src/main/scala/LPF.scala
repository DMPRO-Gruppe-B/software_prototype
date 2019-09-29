package Effects

import chisel3._
import chisel3.util.Counter
import chisel3.experimental.MultiIOModule

class LPF() extends MultiIOModule {

    val io = IO(
      new Bundle {
        val dataInA     = Input(UInt(16.W))
        val dataOut     = Output(UInt(16.W))
        val outputValid = Output(Bool())
      }
    )
  
    val debug = IO(
      new Bundle {
        val myDebugSignal = Output(Bool())
      }
    )
    io.dataOut := 0.U
    io.outputValid := false.B
    debug.myDebugSignal := false.B

    val sample = io.dataInA

    sample := sample * 4
    sample := sample * 3

    sample := sample / 4
    sample := sample / 3


    io.dataOut := sample


}