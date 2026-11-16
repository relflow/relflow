#import "@preview/cetz:0.4.2"

// Canopy columns in native centimeters, from left to right.
// Each row holds the bottom, division centers, and top. A shared gap around
// every division keeps vertical spacing uniform while preserving the outer tips.
#let columns = (
  (1.374857785581, 1.564099268585, 1.674918593783),
  (1.293967047987, 1.64, 1.831893292710),
  (1.215364726657, 1.3, 1.841448444713),
  (1.189882937103, 1.52, 1.8, 1.929998438214),
  (1.182897392128, 1.36, 1.896883877961),
  (1.199593717299, 1.45, 1.889944691743),
  (1.222819338925, 1.64, 1.802300406981),
  (1.262573950573, 1.871375402795),
  (1.252553580004, 1.52, 1.8, 2.068873243893),
  (1.146575053791, 1.36, 1.86, 2.094118772957),
  (1.055859578756, 1.45, 1.9, 2.174612103756),
  (1.019667013501, 1.64, 2.145376598501),
  (1.070858188396, 1.3, 1.528763882958, 1.98, 2.157553792521),
  (1.064575466046, 1.591609496289, 1.8, 2.043881960282),
  (1.203661278336, 1.538896769574, 1.893947405344, 2.016245683181),
  (1.68, 1.90),
)

#let canopy(ink, width: .106, radius: .022, gap: .052) = {
  import cetz.draw: *
  for (index, stops) in columns.enumerate() {
    let x = .281 + index * .140
    for part in range(stops.len() - 1) {
      let low = stops.at(part) + if part == 0 { 0 } else { gap / 2 }
      let high = stops.at(part + 1) - if part == stops.len() - 2 { 0 } else { gap / 2 }
      rect((x - width / 2, low), (x + width / 2, high),
        radius: calc.min(radius, width / 2, (high - low) / 2),
        fill: ink, stroke: none)
    }
  }
}
