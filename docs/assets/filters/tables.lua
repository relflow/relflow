function Table(table)
  if not quarto.doc.is_format("html") then
    return
  end

  local caption = pandoc.utils.stringify(table.caption.long)
  return pandoc.Div({table}, pandoc.Attr("", {"table-responsive"}, {
    tabindex = "0",
    role = "region",
    ["aria-label"] = caption ~= "" and caption or "Scrollable table"
  }))
end
