/* === analysis.js (Wired to the Live API) === */
document.addEventListener("DOMContentLoaded", () => {

    // --- 1. GET ALL HTML ELEMENTS ---
    const niftySelect = document.getElementById("niftySelect");
    const generateBtn = document.getElementById("generateBtn");
    const placeholderCard = document.getElementById("placeholderCard");
    const fullWidthCharts = document.getElementById("fullWidthCharts");
    const statsCard = document.getElementById("statsCard");

    // News
    const newsTitle = document.getElementById("newsTitle");
    const newsList = document.getElementById("newsList");
    
    // Stats
    const statsTable = document.getElementById("statsTable");

    // Charts
    const predCtx = document.getElementById("predChart");
    const histCtx = document.getElementById("histChart");

    // Grid
    const predGrid = document.getElementById("predGrid");

    // Chart instances (to destroy them before re-drawing)
    let histChartObj;
    let predChartObj;

    // --- 2. DEFINE COMPANIES ---
    // The "value" MUST be the ticker that yfinance understands.
    // For Indian stocks, this is ".NS".
    const companies = [
        "ADANIENT.NS - Adani Enterprises Limited",
        "ADANIPORTS.NS - Adani Ports & SEZ Limited",
        "APOLLOHOSP.NS - Apollo Hospitals Enterprises Limited",
        "ASIANPAINT.NS - Asian Paints Limited",
        "AXISBANK.NS - Axis Bank Limited",
        "BAJAJ-AUTO.NS - Bajaj Auto Limited",
        "BAJAJFINSV.NS - Bajaj Finserv Limited",
        "BAJFINANCE.NS - Bajaj Finance Limited",
        "BEL.NS - Bharti Electronics Limited",
        "BHARTIARTL.NS - Bharti Airtel Limited",
        "CIPLA.NS - Cipla Limited",
        "COALINDIA.NS - Coal India Limited",
        "DRREDDY.NS - Dr. Reddy's Laboratories Limited",
        "EICHERMOT.NS - Eicher Motors Limited",
        "ETERNAL.NS - Eternal Life Insurance Limited",
        "GRASIM.NS - Grasim Industries Limited",
        "HCLTECH.NS - HCL Technologies Limited",
        "HDFCBANK.NS - HDFC Bank Limited",
        "HDFCLIFE.NS - HDFC Life Insurance Company Limited",
        "HINDALCO.NS - Hindalco Industries Limited",
        "HINDUNILVR.NS - Hindustan Unilever Limited",
        "ICICIBANK.NS - ICICI Bank Limited",
        "INDIGO.NS - InterGlobe Aviation Limited",
        "INFY.NS - Infosys Limited",
        "ITC.NS - ITC Limited",
        "JIOFIN.NS - Jio Financial Services Limited",
        "JSWSTEEL.NS - JSW Steel Limited",
        "KOTAKBANK.NS - Kotak Mahindra Bank Limited",
        "LT.NS - Larsen & Toubro Limited",
        "M&M.NS - Mahindra & Mahindra Limited",
        "MARUTI.NS - Maruti Suzuki India Limited",
        "MAXHEALTH.NS - Max Healthcare Institute Limited",
        "NESTLEIND.NS - Nestle India Limited",
        "NTPC.NS - NTPC Limited",
        "ONGC.NS - Oil & Natural Gas Corporation Limited",
        "POWERGRID.NS - Power Grid Corporation of India Limited",
        "RELIANCE.NS - Reliance Industries Limited",
        "SBILIFE.NS - SBI Life Insurance Company Limited",
        "SBIN.NS - State Bank of India",
        "SHRIRAMFIN.NS - Shriram Finance Limited",
        "SUNPHARMA.NS - Sun Pharmaceuticals Industries Limited",
        "TATACONSUM.NS - Tata Consumer Products Limited",
        "TATASTEEL.NS - Tata Steel Limited",
        "TCS.NS - Tata Consultancy Services Limited",
        "TECHM.NS - Tech Mahindra Limited",
        "TITAN.NS - Titan Company Limited",
        "TMPV.NS - Tata Motors Passenger Vehicles Limited",
        "TRENT.NS - TRENT Limited",
        "ULTRACEMCO.NS - UltraTech Cement Limited",
        "WIPRO.NS - Wipro Limited",
    ];

    // --- 3. INITIALIZE THE PAGE ---
    
    // Populate the dropdown
    companies.forEach(c => {
        const opt = document.createElement("option");
        // Get just the ticker (e.g., "ADANIENT.NS")
        const ticker = c.split(" - ")[0]; 
        opt.value = ticker; // The value is now just the ticker
        opt.textContent = c; // The text is the full name
        niftySelect.appendChild(opt);
    });

    // Attach the click listener
    generateBtn.addEventListener("click", generateInsights);

    // --- 4. THIS IS THE NEW "HEART" OF THE SCRIPT ---
    async function generateInsights() {
        const ticker = niftySelect.value;
        if (!ticker) {
            alert("Please select a company first!");
            return;
        }

        // 1. Set loading state
        generateBtn.disabled = true;
        generateBtn.textContent = "Generating...";
        placeholderCard.style.display = "none";
        statsCard.style.display = "block";
        fullWidthCharts.style.display = "block"; // Show the section

        try {
            // 2. Call your real Django API
            const response = await fetch(`/api/predict/${ticker}/`);
            if (!response.ok) {
                // If API returns an error, show it
                const errorData = await response.json();
                throw new Error(errorData.error || "Failed to fetch data");
            }
            
            const data = await response.json(); // This is your REAL JSON!

            // 3. Populate all the UI elements with REAL data
            
            // Populate News
            newsTitle.textContent = `Latest News (${data.ticker})`;
            newsList.innerHTML = ""; // Clear old news
            if (data.latest_news.length > 0) {
                data.latest_news.forEach(n => {
                    const div = document.createElement("div");
                    div.className = "news-item"; // Using a class from your HTML
                    let sentimentClass = "";
                    if (n.sentiment_label === "Positive") {
                        sentimentClass = "sent-positive";
                    } else if (n.sentiment_label === "Negative") {
                        sentimentClass = "sent-negative";
                    } else {
                        sentimentClass = "sent-neutral";
                    }
                    div.innerHTML = `
                        <a href="${n.url}" target="_blank">${n.title}</a>
                        <div class="muted">
                            ${n.source} • ${n.published_at || "—"}
                        </div>
                        <div class="sentiment-badge ${sentimentClass}">
                            ${n.sentiment_label} | ${n.sentiment_score}
                        </div>
                    `;
                    newsList.appendChild(div);
                });
            } else {
                newsList.innerHTML = `<div class="muted" style="margin-bottom:8px;">• No recent news found.</div>`;
            }

            // Populate Stats
            statsTable.innerHTML = ""; // Clear old stats
            // We use key_stats from your API
            const entries = Object.entries(data.key_stats);
            const leftColumn = entries.slice(0, 7);
            const rightColumn = entries.slice(7);

            statsTable.innerHTML = "";

            // Create two containers
            const leftDiv = document.createElement("div");
            const rightDiv = document.createElement("div");

            [leftColumn, rightColumn].forEach((column, index) => {
                const container = index === 0 ? leftDiv : rightDiv;

                column.forEach(([k, v]) => {
                    let formattedValue = v;

                    if (typeof v === 'number') {

                        if (k === 'volume' || k === 'market_cap') {
                            formattedValue = v.toLocaleString();
                        }
                        else if (k === 'pe_ratio' || k === 'forward_pe' || k === 'beta' || k === 'eps_basic') {
                            formattedValue = v.toFixed(2);
                        }
                        else {
                            formattedValue = `₹${v.toFixed(2)}`;
                        }
                    }

                    const row = document.createElement("div");
                    row.className = "stats-row";

                    row.innerHTML = `
                        <span class="stats-label">
                            ${k.replace("_", " ").toUpperCase()}
                        </span>
                        <span class="stats-value">
                            ${formattedValue}
                        </span>
                    `;

                    container.appendChild(row);
                });
            });

            statsTable.appendChild(leftDiv);
            statsTable.appendChild(rightDiv);

            // Populate Historical Chart
            if (window.histChartObj) window.histChartObj.destroy();
            window.histChartObj = new Chart(histCtx, {
                type: "line",
                data: {
                    // Use REAL dates from the API
                    labels: data.historical_graph_data.dates, 
                    datasets: [{
                        label: `${data.ticker} (Past 30 Days)`,
                        // Use REAL prices from the API
                        data: data.historical_graph_data.prices, 
                        borderWidth: 2,
                        borderColor: "#00E0FF",
                        tension: 0.3
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: "white" } } }, scales: { x: { ticks: { color: "rgba(255,255,255,0.6)" } }, y: { ticks: { color: "rgba(255,255,255,0.6)" } } } }
            });

            // Populate Prediction Chart
            if (window.predChartObj) window.predChartObj.destroy();
            window.predChartObj = new Chart(predCtx, {
                type: "line",
                data: {
                    // Use REAL predicted dates from the API
                    labels: data.prediction_graph_data.dates, 
                    datasets: [{
                        label: `Predicted Price (₹)`,
                        // Use REAL predicted prices from the API
                        data: data.prediction_graph_data.prices, 
                        borderWidth: 2,
                        borderColor: "#14FFEC",
                        tension: 0.3,
                        fill: false
                    }]
                },
                options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: "white" } } }, scales: { x: { ticks: { color: "rgba(255,255,255,0.6)" } }, y: { ticks: { color: "rgba(255,255,255,0.6)" } } } }
            });

            // Populate Predicted Values Grid
            predGrid.innerHTML = ""; // Clear old grid
            data.prediction_graph_data.dates.forEach((date, i) => {
                const price = data.prediction_graph_data.prices[i];
                const div = document.createElement("div");
                div.className = "pred-item"; // From your old JS
                div.innerHTML = `<div class="pred-date">${date}</div><div class="pred-price">₹${price.toFixed(2)}</div>`;
                predGrid.appendChild(div);
            });

            // =========================
            // RISK + SIGNAL SECTION
            // =========================

            const prices = data.historical_graph_data.prices;
            const predictedPrices = data.prediction_graph_data.prices;

            if (!prices.length || !predictedPrices.length) {
                console.warn("No price data available");
                return;
            }
            const lastPrice = prices[prices.length - 1];
            const predictedFirst = predictedPrices[0];

            // RISK SCORE CALCULATION

            let volatility = 0;
            for (let i = 1; i < prices.length; i++) {
                volatility += Math.abs(prices[i] - prices[i - 1]);
            }
            volatility = volatility / prices.length;

            // Normalize risk score (simple scaling)
            let riskScore = Math.min(100, (volatility / lastPrice) * 1000);

            const riskFill = document.getElementById("riskFill");
            const riskLabel = document.getElementById("riskLabel");
            riskFill.style.width = riskScore + "%";
            if (riskScore < 30) {
                riskLabel.textContent = "Low Risk";
                riskFill.style.boxShadow = "0 0 15px #14FFEC";
            }
            else if (riskScore < 60) {
                riskLabel.textContent = "Moderate Risk";
                riskFill.style.boxShadow = "0 0 15px #00E0FF";
            }
            else {
                riskLabel.textContent = "High Risk";
                riskFill.style.boxShadow = "0 0 15px #ff6b6b";
            }

            // BUY / SELL / HOLD SIGNAL
            const signalText = document.getElementById("tradeSignalText");
            const signalArrow = document.getElementById("signalArrow");
            const confidenceDiv = document.getElementById("signalConfidence");
            const signalNote = document.getElementById("signalNote");

            signalText.classList.remove("signal-buy", "signal-sell", "signal-hold");

            let confidencePercent = Math.abs((predictedFirst - lastPrice) / lastPrice) * 100;
            confidencePercent = Math.min(confidencePercent, 100).toFixed(2);

            if (predictedFirst > lastPrice * 1.02) {
                signalText.textContent = "BUY";
                signalArrow.textContent = "↑";
                signalText.classList.add("signal-buy");
                signalNote.textContent = "Upward momentum detected.";
            } 
            else if (predictedFirst < lastPrice * 0.98) {
                signalText.textContent = "SELL";
                signalArrow.textContent = "↓";
                signalText.classList.add("signal-sell");
                signalNote.textContent = "Downward pressure expected.";
            } 
            else {
                signalText.textContent = "HOLD";
                signalArrow.textContent = "→";
                signalText.classList.add("signal-hold");
                signalNote.textContent = "Sideways consolidation likely.";
            }

            confidenceDiv.textContent = "Signal Strength: " + confidencePercent + "%";

        } catch (error) {
            console.error("Failed to generate insights:", error);
            alert("Error: " + error.message);
            // Show placeholder again if it fails
            placeholderCard.style.display = "block";
            statsCard.style.display = "none";
            fullWidthCharts.style.display = "none";
        } finally {
            // 4. Reset button
            generateBtn.disabled = false;
            generateBtn.textContent = "Generate Insights";
        }
    }
});