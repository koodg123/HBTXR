package network

import block.BlockSequence
import spinal.core._

object generate_whole_network_verilog extends App {
  val begin_id = -1
  val close_id = 24
  // spinal config
  val config = SpinalConfig(
    defaultConfigForClockDomains = ClockDomainConfig(resetKind = SYNC, resetActiveLevel = LOW)
  )
  // use spinalVerilog to generate verilog
  SpinalVerilog(config)(new BlockSequence(begin_id, close_id)).mergeRTLSource()
}
