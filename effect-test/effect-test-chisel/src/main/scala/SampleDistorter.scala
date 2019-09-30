package Effects

import chisel3._
import chisel3.util.Counter
import chisel3.experimental.MultiIOModule

class SampleDistorter() extends MultiIOModule {

    val io = IO(
      new Bundle {
        val dataInA     = Input(UInt(32.W))
        val dataOut     = Output(UInt(32.W))
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

    io.dataOut := sample
}