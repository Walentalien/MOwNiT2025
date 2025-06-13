// src/crawler.rs
use std::collections::{HashSet, VecDeque};
use std::fs::{create_dir_all, File};
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;
use reqwest::blocking::Client;
use scraper::{Html, Selector};
use url::Url;
use std::path::Path;

const BASE: &str = "https://en.wikipedia.org";
const START: &str = "https://en.wikipedia.org/wiki/Special:Random";
const MAX_PAGES: usize = 10000;
const NUM_THREADS: usize = 8;

pub fn run_crawler() {
    if Path::new("data").read_dir().map(|rd| rd.count()).unwrap_or(0) >= MAX_PAGES {
        println!("Crawler skipped: found existing data/ directory with enough pages.");
        return;
    }

    println!("Starting Wikipedia crawler...");
    create_dir_all("data").unwrap();

    let visited = Arc::new(Mutex::new(HashSet::new()));
    let queue = Arc::new(Mutex::new(VecDeque::from([START.to_string()])));
    let client = Arc::new(Client::builder().timeout(Duration::from_secs(10)).build().unwrap());

    let mut handles = vec![];

    for _ in 0..NUM_THREADS {
        let visited = Arc::clone(&visited);
        let queue = Arc::clone(&queue);
        let client = Arc::clone(&client);

        let handle = thread::spawn(move || {
            while visited.lock().unwrap().len() < MAX_PAGES {
                let url = {
                    let mut q = queue.lock().unwrap();
                    q.pop_front()
                };

                if let Some(url) = url {
                    let mut v = visited.lock().unwrap();
                    if v.contains(&url) { continue; }
                    v.insert(url.clone());
                    drop(v);

                    if let Ok(resp) = client.get(&url).send() {
                        if let Ok(body) = resp.text() {
                            let doc = Html::parse_document(&body);
                            let selector = Selector::parse("a[href]").unwrap();
                            let text_selector = Selector::parse("p").unwrap();

                            let base = Url::parse(BASE).unwrap();
                            let mut q = queue.lock().unwrap();

                            for elem in doc.select(&selector) {
                                if let Some(link) = elem.value().attr("href") {
                                    if link.starts_with("/wiki/") && !link.contains(":") {
                                        if let Ok(next_url) = base.join(link) {
                                            let next = next_url.to_string();
                                            if !visited.lock().unwrap().contains(&next) {
                                                q.push_back(next);
                                            }
                                        }
                                    }
                                }
                            }

                            let mut text = String::new();
                            for elem in doc.select(&text_selector) {
                                text += &elem.text().collect::<Vec<_>>().join(" ");
                                text += "\n";
                            }

                            let safe_name = url.replace(BASE, "").replace("/wiki/", "").replace('/', "_");
                            let filename = format!("data/page_{}.txt", safe_name);
                            if let Ok(mut file) = File::create(&filename) {
                                let _ = file.write_all(text.as_bytes());
                            }
                        }
                    }
                } else {
                    thread::sleep(Duration::from_millis(100));
                }
            }
        });

        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }

    println!("Scraping complete.");
}
