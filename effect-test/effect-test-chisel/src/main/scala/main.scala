package Effects
import chisel3._
import java.io.File
import chisel3.iotesters.PeekPokeTester
import org.scalatest.{Matchers, FlatSpec}
import scala.io.Source
import java.io._


object main {
  def main(args: Array[String]): Unit = {
    val s = """
    | Attempting to "run" a chisel program is rather meaningless.
    | Instead, try running the tests, for instance with "test" or "testOnly Examples.MyIncrementTest
    | 
    | If you want to create chisel graphs, simply remove this message and comment in the code underneath 
    | to generate the modules you're interested in.
    """.stripMargin
    println(s)

    chisel3.iotesters.Driver.execute(args,() => new SampleDistorter()) { c =>
      new SampleDistorterTest(c)
    }
  }

  class SampleDistorterTest(c: SampleDistorter) extends PeekPokeTester(c) {
    println("Reading from file")
    val samplefile = "fileopen.scala"

    val inputFile = new File("./../inputsamples.txt")
    val outputFile = new File("./../outputsamples.txt")

    val pwClear = new PrintWriter(new FileOutputStream(outputFile),false)
    pwClear.write("")
    pwClear.close()
    val pw = new PrintWriter(new FileOutputStream(outputFile),true)

    for (line <- io.Source.fromFile(inputFile).getLines){
      val sample = (line.toInt)
      poke(c.io.dataInA,sample)
      pw.write(peek(c.io.dataOut).toString + "\n")
      step(1)
    }
    pw.close()
  }
}