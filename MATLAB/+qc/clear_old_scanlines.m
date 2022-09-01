function clear_old_scanlines()

global plsdata



while true
  try
    plsdata.daq.inst.card.dropNextScanline();
  catch ME
    if ME.ExceptionObject{1} == py.type(py.RuntimeError)
      % all scanlines blocked
      break
    else
      % unkown exception
      rethrow(ME);
    end
  end
end