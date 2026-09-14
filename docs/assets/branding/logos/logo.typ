#import "@preview/cetz:0.4.2"

#let palettes = (
  light: (
    background: rgb("#f2f0e6"), ink: rgb("#234631"),
    canopy: (rgb("#a5b989"), rgb("#afc191"), rgb("#b8c998")),
  ),
  dark: (
    background: rgb("#192d24"), ink: rgb("#dce5ce"),
    canopy: (rgb("#465f4b"), rgb("#4b634f"), rgb("#506853")),
  ),
)

// Native CeTZ coordinates: centimeters, with positive y upward.
// Subdivision preserves the continuous contours at their shared joins.
#let split(curve, t) = {
  let (start, end, first, second) = curve
  let mix(a, b) = (a.at(0) * (1 - t) + b.at(0) * t, a.at(1) * (1 - t) + b.at(1) * t)
  let a = mix(start, first)
  let b = mix(first, second)
  let c = mix(second, end)
  let d = mix(a, b)
  let e = mix(b, c)
  let joint = mix(d, e)
  ((start, joint, a, d), (joint, end, e, c))
}

// One uninterrupted curve carries the f from its upright stem into the upper shoot.
#let f-outer = split(((1.39, .61), (2.11, 1.91), (1.39, 1.46), (1.39, 2.04)), .38)
// The lower shoot eases directly out of the slimmer stem and finishes horizontally.
#let f-lower = split(((1.58, .61), (1.995, 1.235), (1.58, .94), (1.58, 1.235)), .58)
// Retain the r's shoulder and nub, blending only the last part of each stem edge.
#let r-left = split(((1, .61), (1.01, 1.17), (1.02, .8), (1.04, .99)), .1).last()
#let r-right = split(((1.17, 1.19), (1.17, .61), (1.22, .99), (1.16, .8)), .65).first()

// Each pair of contours runs from the stem toward its rounded tip.
#let contours = (
  (
    upper: ((1.17, 1.19), (0.46, 1.52), (1.11, 1.47), (0.79, 1.68)),
    lower: ((1.01, 1.17), (0.48, 1.46), (0.97, 1.38), (0.75, 1.55)),
    tip: (
      (0.440711121, 1.478076846),
      (0.4728, 1.4576), (0.450114974, 1.449865288),
      (0.431417255, 1.505958446), (0.452607285, 1.516415653),
    ),
  ),
  (
    upper: f-outer.last(),
    // Keep the inner edge tapering through the bend, without a narrow neck.
    lower: ((1.56, 1.45), (2.1, 1.825), (1.6, 1.79), (1.8, 1.88)),
    tip: (
      (2.147101917, 1.859839804),
      (2.120106922, 1.908175139), (2.152095486, 1.902285138),
      (2.142079814, 1.817151931), (2.109737036, 1.823214877),
    ),
  ),
  (
    upper: ((1.57, 1.16), (2.005, 1.3), (1.715, 1.265), (1.86, 1.3)),
    lower: f-lower.last(),
    tip: (
      (2.032882366, 1.2675),
      (2.012891768, 1.3), (2.037866416, 1.299896323),
      (2.027867229, 1.234901609), (2.002579045, 1.235),
    ),
  ),
)

// Two curves close each shoot with matching tangent and curvature at all three joins.
// The tip data stores the midpoint, then the two controls on either side of it.
#let tip(incoming, outgoing, controls) = {
  import cetz.draw: bezier
  let (middle, first, second, third, fourth) = controls
  bezier(incoming.at(1), middle, first, second)
  bezier(middle, outgoing.at(1), third, fourth)
}

// Continuous contours taper into the foliage; the f's lower arm grows from its spine.
#let branches(ink) = {
  import cetz.draw: *
  set-style(stroke: none, fill: ink)
  let (left, upper, lower) = contours
  scope({
    translate((.06, 0))
    merge-path(close: true, {
      // Match tangent and curvature at the splices; enter each foot vertically.
      bezier((1, .55), r-left.first(), (1, .613486), (1, .609046))
      bezier(..r-left)
      bezier(..left.lower)
      tip(left.lower, left.upper, left.tip)
      let (start, end, first, second) = left.upper
      bezier(end, start, second, first)
      bezier(..r-right)
      bezier(r-right.at(1), (1.17, .55), (1.17, .725231), (1.17, .710425))
      bezier((1.17, 0.55), (1.14, 0.52), (1.17, 0.5334), (1.1566, 0.52))
      line((1.14, 0.52), (1.03, 0.52))
      bezier((1.03, 0.52), (1, 0.55), (1.0134, 0.52), (1, 0.5334))
    })
    // A short rounded stem terminal gives the mirrored r its shoulder.
    merge-path(close: true, {
      bezier((1.06, 1.15), (1.09, 1.36), (1.11, 1.21), (1.09, 1.3))
      bezier((1.09, 1.36), (1.2, 1.36), (1.09, 1.42), (1.2, 1.42))
      // Meet the trunk at t=.2 with its tangent so the join stays smooth.
      bezier((1.2, 1.36), (1.18824, 1.07112), (1.2, 1.27), (1.18164, 1.16932))
      bezier((1.18824, 1.07112), (1.06, 1.15), (1.15, 1.03), (1.06, 1.09))
    })
  })
  scope({
    translate((-.06, 0))
    merge-path(close: true, {
      bezier(..f-outer.first())
      bezier(..upper.upper)
      tip(upper.upper, upper.lower, upper.tip)
      let (start, end, first, second) = upper.lower
      bezier(end, start, second, first)
      bezier((1.56, 1.45), (1.54, 1.17), (1.553695117, 1.396408492), (1.54, 1.24))
      bezier((1.54, 1.17), (1.57, 1.16), (1.54, 1.135893697), (1.543251164, 1.140630154))
      bezier(..lower.upper)
      tip(lower.upper, lower.lower, lower.tip)
      let (start, end, first, second) = lower.lower
      bezier(end, start, second, first)
      let (start, end, first, second) = f-lower.first()
      bezier(end, start, second, first)
      line((1.58, .61), (1.58, .55))
      bezier((1.58, .55), (1.55, .52), (1.58, .5334), (1.5666, .52))
      line((1.55, .52), (1.42, .52))
      bezier((1.42, 0.52), (1.39, 0.55), (1.4034, 0.52), (1.39, 0.5334))
      line((1.39, 0.55), (1.39, 0.61))
    })
  })
}

// One connected crown rises with the upper shoot and wraps each outward branch tip.
#let crown(palette) = {
  import cetz.draw: *
  set-style(stroke: none)
  // The illuminated edge reuses the silhouette so the colors meet without seams.
  let shoulder = (
    ((.43, 1.84), (.63, 1.82), (.51, 1.86), (.58, 1.84)),
    ((.63, 1.82), (.7, 1.93), (.61, 1.88), (.66, 1.92)),
    ((.7, 1.93), (.9, 1.85), (.78, 1.93), (.86, 1.9)),
    ((.9, 1.85), (1.02, 1.88), (.93, 1.89), (.98, 1.9)),
    ((1.02, 1.88), (1.16, 1.76), (1.07, 1.87), (1.12, 1.8)),
    ((1.16, 1.76), (1.34, 1.95), (1.24, 1.8), (1.26, 1.91)),
    ((1.34, 1.95), (1.49, 2.1), (1.34, 2.04), (1.42, 2.1)),
    ((1.49, 2.1), (1.58, 2.08), (1.53, 2.1), (1.56, 2.09)),
    ((1.58, 2.08), (1.78, 2.2), (1.61, 2.15), (1.73, 2.21)),
    ((1.78, 2.2), (1.77, 2.13), (1.79, 2.18), (1.79, 2.15)),
    ((1.77, 2.13), (2.07, 2.12), (1.88, 2.17), (2.01, 2.17)),
  )
  let lower-shoulder = (
    ((1.98, 1.51), (2.19, 1.51), (2.04, 1.55), (2.14, 1.55)),
    ((2.19, 1.51), (2.13, 1.44), (2.19, 1.48), (2.16, 1.45)),
    ((2.13, 1.44), (2.36, 1.32), (2.23, 1.45), (2.34, 1.4)),
  )
  merge-path(close: true, fill: palette.canopy.at(0), {
    bezier((1.08, 1.22), (.74, 1.2), (.98, 1.2), (.83, 1.16))
    bezier((.74, 1.2), (.45, 1.29), (.64, 1.16), (.51, 1.23))
    bezier((.45, 1.29), (.24, 1.48), (.32, 1.3), (.23, 1.41))
    bezier((.24, 1.48), (.34, 1.55), (.25, 1.51), (.29, 1.54))
    bezier((.34, 1.55), (.26, 1.66), (.29, 1.57), (.26, 1.63))
    bezier((.26, 1.66), (.42, 1.7), (.31, 1.7), (.38, 1.72))
    bezier((.42, 1.7), (.43, 1.84), (.39, 1.74), (.39, 1.81))
    for curve in shoulder { bezier(..curve) }
    bezier((2.07, 2.12), (2.02, 2.02), (2.07, 2.08), (2.05, 2.04))
    bezier((2.02, 2.02), (2.29, 1.98), (2.12, 2.07), (2.23, 2.04))
    bezier((2.29, 1.98), (2.23, 1.89), (2.29, 1.95), (2.26, 1.91))
    bezier((2.23, 1.89), (2.39, 1.8), (2.3, 1.89), (2.37, 1.85))
    bezier((2.39, 1.8), (2.24, 1.65), (2.38, 1.72), (2.32, 1.66))
    bezier((2.24, 1.65), (2.05, 1.63), (2.19, 1.64), (2.13, 1.66))
    bezier((2.05, 1.63), (1.98, 1.51), (1.99, 1.6), (1.91, 1.5))
    for curve in lower-shoulder { bezier(..curve) }
    bezier((2.36, 1.32), (2.2, 1.19), (2.34, 1.25), (2.25, 1.2))
    bezier((2.2, 1.19), (2.12, 1.07), (2.2, 1.13), (2.16, 1.07))
    bezier((2.12, 1.07), (1.98, 1.09), (2.06, 1.05), (2.01, 1.06))
    bezier((1.98, 1.09), (1.75, 1.04), (1.92, 1.02), (1.81, 1.0))
    bezier((1.75, 1.04), (1.56, 1.14), (1.67, 1.04), (1.62, 1.1))
    bezier((1.56, 1.14), (1.34, 1.27), (1.48, 1.16), (1.43, 1.27))
    bezier((1.34, 1.27), (1.08, 1.22), (1.25, 1.28), (1.19, 1.22))
  })
  merge-path(close: true, fill: palette.canopy.at(2), {
    for curve in shoulder { bezier(..curve) }
    bezier((2.07, 2.12), (1.88, 1.98), (2.0, 2.04), (1.99, 1.97))
    bezier((1.88, 1.98), (1.69, 1.99), (1.8, 1.96), (1.78, 2.03))
    bezier((1.69, 1.99), (1.53, 1.87), (1.6, 1.98), (1.62, 1.84))
    bezier((1.53, 1.87), (1.38, 1.77), (1.45, 1.9), (1.44, 1.81))
    bezier((1.38, 1.77), (1.19, 1.62), (1.3, 1.73), (1.3, 1.62))
    bezier((1.19, 1.62), (1.03, 1.73), (1.11, 1.61), (1.12, 1.71))
    bezier((1.03, 1.73), (.88, 1.66), (.93, 1.76), (.98, 1.64))
    bezier((.88, 1.66), (.71, 1.75), (.8, 1.67), (.82, 1.77))
    bezier((.71, 1.75), (.56, 1.66), (.61, 1.73), (.65, 1.65))
    bezier((.56, 1.66), (.43, 1.84), (.47, 1.68), (.47, 1.78))
  })
  merge-path(close: true, fill: palette.canopy.at(1), {
    bezier((.35, 1.48), (.61, 1.61), (.39, 1.56), (.51, 1.65))
    bezier((.61, 1.61), (.78, 1.55), (.7, 1.61), (.71, 1.52))
    bezier((.78, 1.55), (.96, 1.46), (.86, 1.58), (.96, 1.53))
    bezier((.96, 1.46), (.76, 1.39), (.94, 1.38), (.84, 1.44))
    bezier((.76, 1.39), (.56, 1.37), (.67, 1.42), (.64, 1.34))
    bezier((.56, 1.37), (.35, 1.48), (.46, 1.37), (.42, 1.45))
  })
  merge-path(close: true, fill: palette.canopy.at(1), {
    for curve in lower-shoulder { bezier(..curve) }
    bezier((2.36, 1.32), (2.15, 1.32), (2.28, 1.34), (2.25, 1.28))
    bezier((2.15, 1.32), (1.98, 1.38), (2.07, 1.35), (2.08, 1.41))
    bezier((1.98, 1.38), (1.82, 1.31), (1.9, 1.35), (1.91, 1.28))
    bezier((1.82, 1.31), (1.7, 1.38), (1.75, 1.32), (1.76, 1.4))
    bezier((1.7, 1.38), (1.98, 1.51), (1.78, 1.45), (1.89, 1.46))
  })
}

#let logo(theme: "light", size: 195pt, tile: true) = {
  let palette = palettes.at(theme)
  let factor = size / 2.6cm * 100%
  std.scale(x: factor, y: factor, reflow: true, cetz.canvas({
    import cetz.draw: *
    // A hidden frame preserves the composition when exporting without a tile.
    rect((0, 0), (2.6, 2.6), radius: .38, fill: if tile { palette.background } else { none }, stroke: none)
    crown(palette)
    branches(palette.ink)
  }))
}

#set page(width: auto, height: auto, margin: 0pt, fill: none)
#logo(theme: sys.inputs.at("theme", default: "light"))
