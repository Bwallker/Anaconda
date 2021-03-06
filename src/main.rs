pub(crate) mod lexer;
pub(crate) mod parser;
pub(crate) mod generate_bytecode;
pub(crate) mod util;
pub(crate) mod runtime;
use clap::{Args, Parser, Subcommand};
use color_eyre::eyre::eyre;
use runtime::{bytecode::BytecodeInterpreter, gc::GarbageCollector};
use generate_bytecode::generate_bytecode;
use parser::parse;
use std::fs::read_to_string;
use std::io::Write;
use std::path::PathBuf;
use std::process::exit;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

const TEST_PROGRAM: &str = "
z += y += x = 1
println(x)
println(y)
println(z)
";

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Commands,
}
#[derive(Subcommand, Debug)]
enum Commands {
    #[clap(alias = "l")]
    Lex(Lex),

    #[clap(alias = "p")]
    Parse(Parse),

    #[clap(alias = "r")]
    Run(Run),

    #[clap(alias = "rt")]
    RunTest,
}
#[derive(Args, Debug)]
struct Lex {
    /// The path to the program to lex. Mutually exclusive with program.
    #[clap(long = "path", parse(from_os_str), value_name = "PATH")]
    program_path: Option<PathBuf>,
    /// The program to lex. Mutually exclusive with program_path.
    #[clap(short, long, value_name = "PROGRAM")]
    program: Option<String>,
}

#[derive(Args, Debug)]
struct Parse {
    /// The path to the program to parse. Mutually exclusive with program.
    #[clap(long = "path", parse(from_os_str), value_name = "PATH")]
    program_path: Option<PathBuf>,
    /// The program to parse. Mutually exclusive with program_path.
    #[clap(short, long, value_name = "PROGRAM")]
    program: Option<String>,
}

#[derive(Args, Debug)]
struct Run {
    /// The path to the program to run. Mutually exclusive with program.
    #[clap(long = "path", parse(from_os_str), value_name = "PATH")]
    program_path: Option<PathBuf>,
    /// The program to run. Mutually exclusive with program_path.
    #[clap(short, long, value_name = "PROGRAM")]
    program: Option<String>,
}

fn main() {
    match run() {
        Ok(()) => (),
        Err(e) => {
            let mut stdout = StandardStream::stdout(ColorChoice::Always);
            stdout
                .set_color(ColorSpec::new().set_fg(Some(Color::Red)))
                .unwrap();
            writeln!(&mut stdout, "Error!").unwrap();
            writeln!(&mut stdout, "------------").unwrap();
            writeln!(&mut stdout, "{e}").unwrap();
            exit(69);
        }
    }
}

fn run() -> color_eyre::Result<()> {
    let args = match Cli::try_parse() {
        Ok(v) => v,
        Err(e) => {
            e.print()?;
            return Err(eyre!(""));
        }
    };
    match args.command {
        Commands::Lex(l) => {
            let contents = match (l.program, l.program_path) {
                (None, None) => {
                    return Err(eyre!(
                    "You must provide either a program or a program path for the lex subcommand."
                ))
                }
                (Some(v), None) => v,
                (None, Some(v)) => read_to_string(v)?,
                (Some(_), Some(_)) => {
                    return Err(eyre!(
                    "You cannot provide both a program and a program path to the lex subcommand."
                ))
                }
            };
            let mut lexer = lexer::lex::Lexer::new(&contents);
            let tokens = lexer.collect_tokens();
            match tokens {
                Ok(v) => println!("{v:#?}"),
                Err(e) => return Err(eyre!(format!("{e}"))),
            }
        }

        Commands::Parse(p) => {
            let contents = match (p.program, p.program_path) {
                (None, None) => {
                    return Err(eyre!(
                    "You must provide either a program or a program path for the parse subcommand."
                ))
                }
                (Some(v), None) => v,
                (None, Some(v)) => read_to_string(v)?,
                (Some(_), Some(_)) => {
                    return Err(eyre!(
                    "You cannot provide both a program and a program path to the parse subcommand."
                ))
                }
            };
            let ast = parse(&contents);
            match ast {
                Ok(mut v) => {
                    let base_block = v.program.base_block.clone();
                    let mut gc = GarbageCollector::new();
                    let bytecode = generate_bytecode(&mut v, &mut gc);
                    v.program.base_block = base_block;
                    std::fs::write("ast.txt", format!("{v:#?}"))?;
                    std::fs::write("bytecode.txt", format!("{bytecode}"))?;
                }
                Err(e) => return Err(eyre!(format!("{e}"))),
            }
        }
        Commands::Run(r) => {
            let contents = match (r.program, r.program_path) {
                (None, None) => {
                    return Err(eyre!(
                    "You must provide either a program or a program path for the run subcommand."
                ))
                }
                (Some(v), None) => v,
                (None, Some(v)) => read_to_string(v)?,
                (Some(_), Some(_)) => {
                    return Err(eyre!(
                    "You cannot provide both a program and a program path to the run subcommand."
                ))
                }
            };
            let ast = parse(&contents);
            let mut ast = match ast {
                Ok(v) => v,
                Err(e) => return Err(eyre!(format!("{e}"))),
            };
            let mut gc = GarbageCollector::new();
            let bytecode = generate_bytecode(&mut ast, &mut gc);
            let mut bytecode_interpreter = BytecodeInterpreter::new(ast.program, bytecode, gc);
            bytecode_interpreter.interpret_bytecode();
        }

        Commands::RunTest => {
            let ast = parse(TEST_PROGRAM);
            let mut ast = match ast {
                Ok(v) => v,
                Err(e) => return Err(eyre!(format!("{e}"))),
            };
            let mut gc = GarbageCollector::new();
            let bytecode = generate_bytecode(&mut ast, &mut gc);
            let mut bytecode_interpreter = BytecodeInterpreter::new(ast.program, bytecode, gc);
            bytecode_interpreter.interpret_bytecode();
        }

    }
    Ok(())
}
