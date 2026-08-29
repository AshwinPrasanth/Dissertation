use crate::{vcsolve, Formula, Graph, Hypergraph, MaxPre, Presolve};
use std::env;
use gag::Gag;

pub enum Task {
    DominatingSet,
    HittingSet,
}

struct Settings {
    maxpre: bool,
    maxpre_techniques: String,
    maxpre_time: f64,
    prune: bool,
}

pub fn solve(task: Task) {
    let mut h = match task {
        Task::DominatingSet => Hypergraph::new_from_graph(Graph::new_from_stdin()),
        Task::HittingSet => Hypergraph::new_from_stdin(),
    };

    let settings = match task {
        Task::DominatingSet => Settings {
            maxpre: true,
            maxpre_techniques: "[bu]#[buvsrglehtG]".to_owned(),
            maxpre_time: 1e20,
            prune: true,
        },
        Task::HittingSet => Settings {
            maxpre: false,
            maxpre_techniques: "[bu]#[buvsrglehtG]".to_owned(),
            maxpre_time: 1e20,
            prune: true,
        },
    };

    // Apply Hitting Set Presolve.
    let mut presolve = Presolve::from_edge_list(h.n, &h.sets);
    presolve.apply(None);
    h.sets = presolve.get_remaining_edges();
    let mut picked: Vec<_> = presolve.get_picks();

    let (g, mapping) = if settings.prune {
        h.prune()
    } else {
        (h.clone(), (0..h.n + 1).collect::<Vec<usize>>())
    };

    if let Ok(debug_instance) = env::var("DEBUG_REDUCED") {
    use std::io::Write;

    let mut file = std::fs::File::create(
        format!("/tmp/{}_reduced.hgr", debug_instance)
    ).unwrap();

    writeln!(
        file,
        "p hs {} {}",
        g.n,
        g.sets.len()
    ).unwrap();

    for edge in &g.sets {
        writeln!(
            file,
            "{}",
            edge.iter()
                .map(|v| v.to_string())
                .collect::<Vec<_>>()
                .join(" ")
        ).unwrap();
    }

    let mut map_file = std::fs::File::create(
        format!("/tmp/{}_mapping.txt", debug_instance)
    ).unwrap();

    for (reduced, original) in mapping.iter().enumerate() {
        writeln!(
            map_file,
            "{} {}",
            reduced,
            original
        ).unwrap();
    }
}

    let mapping_u32: Vec<u32> = mapping.iter().map(|&x| x as u32).collect();
    let instance_name = env::var("INSTANCE").expect("INSTANCE not set");
    let feature_path = env::var("VERTEX_FEATURES").expect("VERTEX_FEATURES not set");

    let feature_file = std::fs::read_to_string(feature_path)
        .expect("Failed to read vertex_features.csv");

    let mut static_features = vec![0.0f64; (h.n + 1) * 12];

    for line in feature_file.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();

        if fields.len() != 14 || fields[0] != instance_name {
            continue;
        }

        let vertex: usize = fields[1].parse().unwrap();

        if vertex > h.n {
            continue;
        }

        for j in 0..12 {
            static_features[vertex * 12 + j] =
                fields[j + 2].parse::<f64>().unwrap();
        }
    }

    // Solve SAT instances with Kissat
    if g.is_sat_instance() {
        println!("[ROUTE] {} SAT/Kissat n={}", instance_name, g.n);
        let mut solver = kissat::Solver::new();
        let mut vars = Vec::new();
        for _ in 0..g.n / 2 {
            vars.push(solver.var());
        }
        for set in g.sets.iter() {
            let clause: Vec<_> = set
                .iter()
                .map(|x| {
                    if x % 2 == 1 {
                        vars[x / 2]
                    } else {
                        !vars[(x - 1) / 2]
                    }
                })
                .collect();
            solver.add(&clause);
        }
        if let Some(solution) = solver.sat() {
            let mut result = picked.clone();
            for x in 0..g.n / 2 {
                match solution.get(vars[x]) {
                    Some(true) => result.push(mapping[x * 2 + 1]),
                    Some(false) => result.push(mapping[x * 2 + 2]),
                    None => (),
                };
            }
            println!("{}", result.len());
            for v in result.iter() {
                println!("{}", v);
            }
            return;
        }
    }

    // Solve VC Instances via a Clique solver
    if g.is_vc_instance() && g.n <= 350 {
        println!("[ROUTE] {} VC solver n={}", instance_name, g.n);
        let mut result = picked.clone();
        let solution = vcsolve::vc_solve(&g);
        for &u in solution.iter() {
            result.push(mapping[u]);
        }
        println!("{}", result.len());
        for v in result.iter() {
            println!("{}", v);
        }
        return;
    }

    // Remaining instances use MaxSAT solver
    println!(
    "[ROUTE] {} CaDiCaL/MaxSAT n={}",
    instance_name,
    g.n
);
    // Build the MaxSAT formula
    let mut hard = Vec::new();
    let mut soft = Vec::new();
    for set in g.sets.iter() {
        hard.push(set.iter().map(|x| *x as i32).collect());
    }
    for v in 1..=g.n {
        soft.push((vec![-(v as i32)], 1));
    }

    // Apply MaxSAT preprocessing
    let mut maxpre = MaxPre::new(hard.clone(), soft.clone());
    let (hard, soft) = match settings.maxpre {
        true => maxpre.run(
            settings.maxpre_techniques.as_str(),
            false,
            settings.maxpre_time,
        ),
        false => (hard, soft),
    };

    // Run solver
    let mut vertex_incidence = vec![Vec::<u32>::new(); g.n + 1];

    for (edge_idx, edge) in g.sets.iter().enumerate() {
        for &v in edge {
            vertex_incidence[v].push(edge_idx as u32);
        }
    }

    let mut edge_data = Vec::new();
    let mut edge_offsets = vec![0usize];

    for v in 0..=g.n {
        edge_data.extend_from_slice(&vertex_incidence[v]);
        edge_offsets.push(edge_data.len());
    }	
   
    let mut phi = Formula::new();
    phi.set_vertex_mapping(
    &mapping_u32,
    &edge_data,
    &edge_offsets,
    &static_features,
);
    let ml_branch = env::var("ML_BRANCH")
        .map(|v| v == "1")
        .unwrap_or(false);

    phi.set_ml_branch(ml_branch);
    for c in hard.iter() {
        phi.add_clause(c, None);
    }
    for (c, w) in soft.iter() {
        phi.add_clause(c, Some(*w as i64));
    }
    let result = {
        let _gag = Gag::stdout().unwrap();
        phi.solve()
    };

    phi.print_stats();

    // Reconstruct and output
    if result {
        let sol = (1..=g.n)
            .map(|v| v as i32)
            .map(|v| match phi.value(v) {
                true => v,
                false => -v,
            })
            .collect::<Vec<i32>>();
        let sol = match settings.maxpre {
            true => maxpre.reconstruct(g.n, sol),
            false => sol,
        };

        for v in sol {
            if v > 0 {
                picked.push(mapping[v as usize]);
            }
        }
        println!("{}", picked.len());
        for v in picked.iter() {
            println!("{}", v);
        }
    }
}
