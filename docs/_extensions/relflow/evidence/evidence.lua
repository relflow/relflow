-- Resolve generated evidence after the project's pre-render step.
return {
  ["evidence"] = function(args)
    local identifier = pandoc.utils.stringify(args[1])
    local section = pandoc.utils.stringify(args[2])
    local sections = {status = true, evidence = true, script = true}
    if not identifier:match("^P%d%d%d$") or not sections[section] then
      error("Use {{< evidence P001 status|evidence|script >}} with a registered evidence ID.")
    end
    local path = quarto.project.directory .. "/evidence/_generated/" .. identifier .. "-" .. section .. ".md"
    local file, reason = io.open(path, "r")
    if not file then
      error("Cannot read evidence fragment " .. path .. ": " .. reason)
    end
    local content = file:read("*a")
    file:close()
    return pandoc.read(content, "markdown").blocks
  end
}
