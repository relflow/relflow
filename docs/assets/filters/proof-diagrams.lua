function Image(image)
  local source = quarto.doc.input_file:gsub("\\", "/")
  if not (source:match("^proofs/") or source:match("/proofs/")) then
    return
  end
  if not quarto.doc.is_format("html") or not image.src:match("assets/diagrams/.+%.svg$") then
    return
  end
  return pandoc.Span({image}, pandoc.Attr("", {"proof-diagram"}, {
    tabindex = "0",
    role = "region",
    ["aria-label"] = "Scrollable proof diagram"
  }))
end
