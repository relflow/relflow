function Image(image)
  local source = quarto.doc.input_file:gsub("\\", "/")
  if not (source:match("^evidence/") or source:match("/evidence/")) then
    return
  end
  if not quarto.doc.is_format("html") or not image.src:match("assets/diagrams/.+%.svg$") then
    return
  end
  return pandoc.Span({image}, pandoc.Attr("", {"evidence-diagram"}, {
    tabindex = "0",
    role = "region",
    ["aria-label"] = "Scrollable evidence diagram"
  }))
end
