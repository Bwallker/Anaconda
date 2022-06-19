mod lexer;
mod parser;

use clap::{Args, Parser, Subcommand};
use color_eyre::eyre::eyre;
use parser::parse;
use std::fs::read_to_string;
use std::io::Write;
use std::path::PathBuf;
use std::process::exit;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use parser::bytecode::generate_bytecode;
const TEST_PROGRAM: &str = "\t ";
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

    #[clap(alias = "lt")]
    LexTestProgram,

    #[clap(alias = "p")]
    Parse(Parse),
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
fn main() {
    match run() {
        Ok(()) => (),
        Err(e) => {
            let mut stdout = StandardStream::stdout(ColorChoice::Always);
            stdout
                .set_color(ColorSpec::new().set_fg(Some(Color::Red)))
                .unwrap();
            writeln!(&mut stdout, "Lexer Error!").unwrap();
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
        Commands::LexTestProgram => {
            let mut lexer = lexer::lex::Lexer::new(TEST_PROGRAM);
            let tokens = lexer.collect_tokens();
            match tokens {
                Ok(v) => println!("{v:#?}"),
                Err(e) => println!("{e}"),
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
            let statements = parse(&contents);
            match statements {
                Ok(v) => {
                    println!("{v:#?}");
                    println!("{:#?}", generate_bytecode(v.statements))
                }
                Err(e) => return Err(eyre!(format!("{e}"))),
            }
        }
    }
    Ok(())
}
