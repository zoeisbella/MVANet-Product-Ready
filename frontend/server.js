const express = require('express');
const path = require('path');
const { exec } = require('child_process');
const http = require('http');

const app = express();
const PORT = 3000;

// Serve static files from the current directory
app.use(express.static(path.join(__dirname)));

// Main route to serve the index.html
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Proxy API requests to the backend
app.use('/predict', (req, res) => {
    const { method, headers, url } = req;
    const backendUrl = `http://localhost:8001${url}`;
    
    // Parse the backend URL
    const parsedUrl = new URL(backendUrl);
    
    const options = {
        hostname: parsedUrl.hostname,
        port: parsedUrl.port || 8001,
        path: parsedUrl.pathname + parsedUrl.search,
        method: method,
        headers: headers,
    };

    const proxyReq = http.request(options, (proxyRes) => {
        let data = '';
        proxyRes.on('data', (chunk) => {
            data += chunk;
        });
        
        proxyRes.on('end', () => {
            // Set appropriate headers and send response
            res.set(proxyRes.headers);
            res.status(proxyRes.statusCode).send(data);
        });
    });

    proxyReq.on('error', (err) => {
        console.error('Proxy error:', err);
        res.status(500).send('Proxy error');
    });

    // Handle request body for POST/PUT requests
    if (method === 'POST' || method === 'PUT') {
        req.on('data', (chunk) => {
            proxyReq.write(chunk);
        });
        
        req.on('end', () => {
            proxyReq.end();
        });
    } else {
        proxyReq.end();
    }
});

// Also proxy the predict_and_download endpoint
app.use('/predict_and_download', (req, res) => {
    const { method, headers, url } = req;
    const backendUrl = `http://localhost:8001${url}`;
    
    // Parse the backend URL
    const parsedUrl = new URL(backendUrl);
    
    const options = {
        hostname: parsedUrl.hostname,
        port: parsedUrl.port || 8001,
        path: parsedUrl.pathname + parsedUrl.search,
        method: method,
        headers: headers,
    };

    const proxyReq = http.request(options, (proxyRes) => {
        let data = '';
        proxyRes.on('data', (chunk) => {
            data += chunk;
        });
        
        proxyRes.on('end', () => {
            res.set(proxyRes.headers);
            res.status(proxyRes.statusCode).send(data);
        });
    });

    proxyReq.on('error', (err) => {
        console.error('Proxy error:', err);
        res.status(500).send('Proxy error');
    });

    if (method === 'POST' || method === 'PUT') {
        req.on('data', (chunk) => {
            proxyReq.write(chunk);
        });
        
        req.on('end', () => {
            proxyReq.end();
        });
    } else {
        proxyReq.end();
    }
});

app.listen(PORT, () => {
    console.log(`Frontend server running at http://localhost:${PORT}`);
    console.log('Please make sure the backend API is running on http://localhost:8001');
});