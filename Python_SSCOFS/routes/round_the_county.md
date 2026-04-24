race: Round the County
direction: counter_clockwise
coordinate_system: WGS84
units:
  angles: degrees
  latlon: decimal_degrees

fixed_marks:
  lydia_shoal_buoy:
    name: "Lydia Shoal buoy G '13' (Q G)"
    lat: 48.599550
    lon: -122.779083
    type: buoy
    passing_rule: leave_to_port

  clements_reef_buoy:
    name: "Clements Reef Buoy R '2'"
    lat: 48.777667
    lon: -122.891733
    type: buoy
    passing_rule: leave_to_port

  danger_shoal_buoy:
    name: "Danger Shoal lighted bell buoy"
    lat: 48.639167
    lon: -123.181000
    type: buoy
    passing_rule: leave_to_port

  patos_lighthouse:
    name: "Patos Island Light"
    lat: 48.789000
    lon: -122.971367
    type: lighthouse_reference
    passing_rule: leave_to_port

  iceberg_point_light:
    name: "Iceberg Point Light 2"
    lat: 48.422050
    lon: -122.894283
    type: lighthouse_reference
    passing_rule: leave_to_port

dynamic_marks:
  saturday_start_rc_boat:
    type: rc_boat
    source: race_committee
    notes: "Start line off Lydia Shoal; exact RC position set on race day."

  saturday_finish_rc_boat:
    type: rc_boat
    source: race_committee
    notes: "Full-course finish line set on race day near Roche Harbor area."

  saturday_finish_orange_pyramid:
    type: buoy
    source: race_committee
    notes: "Paired with saturday_finish_rc_boat for full-course finish line."

  sunday_start_rc_boat:
    type: rc_boat
    source: race_committee
    notes: "Start line off south end of Mosquito Pass; exact RC position set on race day."

  sunday_start_orange_pyramid:
    type: buoy
    source: race_committee
    notes: "Paired with sunday_start_rc_boat for Sunday start line."

  sunday_finish_rc_boat:
    type: rc_boat
    source: race_committee
    notes: "Finish line uses RC boat and Lydia Shoal buoy."

legs:
  leg1_saturday:
    name: "Lydia Shoal to Roche Harbor area"
    start:
      type: line
      approx_region:
        description: "Off Lydia Shoal"
        center_lat: 48.599550
        center_lon: -122.779083
      endpoints:
        - ref: lydia_shoal_buoy
        - ref: saturday_start_rc_boat

    course_constraints:
      - seq: 1
        action: pass_mark
        side: port
        ref: lydia_shoal_buoy

      - seq: 2
        action: keep_feature_to_port
        ref: "Orcas Island"
        feature_type: island_polygon

      - seq: 3
        action: keep_feature_to_port
        ref: "Sisters Islands"
        feature_type: island_polygon

      - seq: 4
        action: keep_feature_to_port
        ref: "Clark Island"
        feature_type: island_polygon

      - seq: 5
        action: keep_feature_to_port
        ref: "Matia Island"
        feature_type: island_polygon

      - seq: 6
        action: keep_feature_to_port
        ref: "Clements Reef"
        feature_type: reef_polygon

      - seq: 7
        action: pass_mark
        side: port
        ref: clements_reef_buoy

      - seq: 8
        action: keep_feature_to_port
        ref: "Patos Island"
        feature_type: island_polygon

      - seq: 9
        action: keep_feature_to_port
        ref: "Waldron Island"
        feature_type: island_polygon

      - seq: 10
        action: keep_feature_to_port
        ref: "Skipjack Island"
        feature_type: island_polygon

      - seq: 11
        action: keep_feature_to_port
        ref: "Stuart Island"
        feature_type: island_polygon

      - seq: 12
        action: pass_mark
        side: port
        ref: danger_shoal_buoy

    finish_options:
      full_course:
        type: line
        endpoints:
          - ref: saturday_finish_rc_boat
          - ref: saturday_finish_orange_pyramid
        approx_regions:
          - description: "off northwest corner of Pearl Island"
          - description: "north of McCracken Point on Henry Island and east of Battleship Island"
        notes: "Exact finish location chosen by RC on race day."

      short_course:
        type: virtual_meridian
        reference: patos_lighthouse
        longitude_deg: -122.971333
        crossing_rule: "finish when crossing the true-north line through this longitude"
        notes: "This is the published Patos short-course finish meridian."

  leg2_sunday:
    name: "Mosquito Pass area to Lydia Shoal"
    start:
      type: line
      approx_region:
        description: "Off south end of Mosquito Pass"
        approx_center_lat: 48.5900
        approx_center_lon: -123.0050
      endpoints:
        - ref: sunday_start_rc_boat
        - ref: sunday_start_orange_pyramid
      notes: "Approx center is only a planning placeholder; actual line is RC-set."

    course_constraints:
      - seq: 1
        action: keep_feature_to_port
        ref: "San Juan Island"
        feature_type: island_polygon

      - seq: 2
        action: keep_feature_to_port
        ref: "Long Island"
        feature_type: island_polygon

      - seq: 3
        action: keep_feature_to_port
        ref: "Lopez Island"
        feature_type: island_polygon

      - seq: 4
        action: keep_feature_to_port
        ref: "Davidson Rock"
        feature_type: hazard_polygon_or_point

      - seq: 5
        action: keep_feature_to_port
        ref: "Kellett Ledge"
        feature_type: hazard_polygon_or_point

      - seq: 6
        action: keep_feature_to_port
        ref: "James Island"
        feature_type: island_polygon

      - seq: 7
        action: keep_feature_to_port
        ref: "Blakely Island"
        feature_type: island_polygon

    finish:
      type: line
      approx_region:
        description: "Lydia Shoal finish area"
        center_lat: 48.599550
        center_lon: -122.779083
      endpoints:
        - ref: sunday_finish_rc_boat
        - ref: lydia_shoal_buoy
      notes: "Leave RC boat to starboard, Lydia Shoal buoy to port."

implementation_notes:
  - "Use pass_mark(side=port) for discrete buoys/lights."
  - "Use keep_feature_to_port for islands, reefs, and ledges; these are side-of-obstacle topological constraints, not waypoint hits."
  - "For routing, represent islands/reefs as polygons from chart data."
  - "Dynamic RC-set marks must be injected from race-day VHF / RC observations."
  - "Do not snap to the approximate Sunday start center; it is only a seed/placeholder for geometry initialization."