# Copyright 2020 Kaggle Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import traceback
from kaggle_environments import environments, evaluate, make, utils


parser = argparse.ArgumentParser(description="Kaggle Simulations")
parser.add_argument(
    "action",
    choices=["list", "evaluate", "run", "step", "load", "http-server"],
    help="List environments. Evaluate many episodes. Run a single episode. Step the environment. Load the environment. Start http server.",
)
parser.add_argument("--environment", type=str,
                    help="Environment to run against.")
parser.add_argument("--debug", type=bool, help="Print debug statements.")
parser.add_argument(
    "--agents", type=str, nargs="*", help="Agent(s) to run with the environment."
)
parser.add_argument(
    "--configuration",
    type=json.loads,
    help="Environment configuration to setup the environment.",
)
parser.add_argument(
    "--steps",
    type=json.loads,
    help="Environment starting states (default=[resetState]).",
)
parser.add_argument(
    "--episodes", type=int, help="Number of episodes to evaluate (default=1)"
)
parser.add_argument(
    "--render",
    type=json.loads,
    help="Response from run, step, or load. Calls environment render. (default={mode='json'})",
)


def render(args, env):
    mode = utils.get(args.render, str, "json", path=["mode"])
    if mode == "human" or mode == "ansi":
        args.render["mode"] = "ansi"
    elif mode == "ipython" or mode == "html":
        args.render["mode"] = "html"
    else:
        args.render["mode"] = "json"
    return env.render(**args.render)


def action_list(args):
    return json.dumps([*environments])


def action_evaluate(args):
    return json.dumps(
        evaluate(
            args.environment, args.agents, args.configuration, args.steps, args.episodes
        )
    )


def action_step(args):
    env = make(args.environment, args.configuration, args.steps, args.debug)
    env.step(env.__get_actions(args.agents))
    return render(args, env)


def action_run(args):
    env = make(args.environment, args.configuration, args.steps, args.debug)
    env.run(args.agents)
    return render(args, env)


def action_load(args):
    env = make(args.environment, args.configuration, args.steps, args.debug)
    return render(args, env)


def action_handler(args):
    args = utils.structify(
        {
            "action": utils.get(args, str, "list", ["action"]),
            "agents": utils.get(args, list, [], ["agents"]),
            "configuration": utils.get(args, dict, {}, ["configuration"]),
            "environment": args.get("environment", None),
            "episodes": utils.get(args, int, 1, ["episodes"]),
            "steps": utils.get(args, list, [], ["steps"]),
            "render": utils.get(args, dict, {"mode": "json"}, ["render"]),
            "debug": utils.get(args, bool, False, ["debug"])
        }
    )

    for index, agent in enumerate(args.agents):
        agent = utils.read_file(agent, agent)
        args.agents[index] = utils.get_last_callable(agent, agent)

    if args.action == "list":
        return action_list(args)

    if args.environment == None:
        return {"error": "Environment required."}

    try:
        if args.action == "http-server":
            return {"error": "Already running a http server."}
        elif args.action == "evaluate":
            return action_evaluate(args)
        elif args.action == "step":
            return action_step(args)
        elif args.action == "run":
            return action_run(args)
        elif args.action == "load":
            return action_load(args)
        else:
            return {"error": "Unknown Action"}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}


def action_http(args):
    from flask import Flask, request, send_from_directory
    import os

    app = Flask(__name__, static_url_path="", static_folder="kaggle_environments")
    
    # 添加靜態文件路由
    @app.route('/kaggle_environments/<path:filename>')
    def static_files(filename):
        return send_from_directory('kaggle_environments', filename)
    
    # 添加環境特定的 JS 文件路由
    @app.route('/kaggle_environments/envs/<env_name>/<filename>')
    def env_files(env_name, filename):
        return send_from_directory(f'kaggle_environments/envs/{env_name}', filename)
    
    # 添加索引頁面
    @app.route('/index')
    @app.route('/home')
    def index():
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>ConnectX AI 控制中心</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white;
            min-height: 100vh;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }
        .header h1 {
            font-size: 2.5em;
            margin: 0;
            background: linear-gradient(45deg, #1EBEFF, #00D4FF);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .card {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(30, 190, 255, 0.3);
        }
        .card h3 {
            color: #1EBEFF;
            margin-top: 0;
            font-size: 1.4em;
        }
        .button {
            background: linear-gradient(45deg, #1EBEFF, #00D4FF);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            margin: 10px 10px 10px 0;
            font-size: 16px;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s ease;
            font-weight: 600;
        }
        .button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(30, 190, 255, 0.4);
        }
        .status {
            background: rgba(0, 255, 100, 0.2);
            padding: 10px;
            border-radius: 8px;
            border-left: 4px solid #00FF64;
            margin: 15px 0;
        }
        .api-demo {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            font-size: 14px;
            margin: 15px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 ConnectX AI 控制中心</h1>
        <p>你的 94% 勝率 AI 模型已準備就緒！</p>
        <div class="status">
            ✅ 服務器運行中 | 🤖 AI 模型已載入 | 🎮 GUI 系統就緒
        </div>
    </div>

    <div class="cards">
        <div class="card">
            <h3>🎮 即時遊戲演示</h3>
            <p>觀看你的 AI 與隨機對手的實時對戰，包含完整的動畫效果和步驟控制。</p>
            <a href="/demo" class="button" target="_blank">🚀 開始演示</a>
            <a href="/new_game" class="button" target="_blank">🔄 新遊戲</a>
        </div>

        <div class="card">
            <h3>🛠️ 互動式 GUI</h3>
            <p>多種播放器選項，從簡單配置到完全自定義的遊戲界面。</p>
            <a href="/player" class="button" target="_blank">🎮 互動配置播放器</a>
            <a href="/kaggle_player" class="button" target="_blank">🌐 Kaggle 原版播放器</a>
            <p style="font-size: 12px; opacity: 0.8;">
                互動播放器：完全可配置 | 原版播放器：預設遊戲數據
            </p>
        </div>

        <div class="card">
            <h3>📊 API 測試</h3>
            <p>通過 RESTful API 直接調用你的 AI 模型，支持多種輸出格式。</p>
            <div class="api-demo">
POST /
{
  "action": "run",
  "environment": "connectx", 
  "agents": ["submission.py", "random"],
  "render": {"mode": "html"}
}
            </div>
            <button class="button" onclick="testAPI()">🧪 測試 API</button>
            <div id="result"></div>
        </div>

        <div class="card">
            <h3>📈 模型信息</h3>
            <p>你的 ConnectX AI 詳細信息：</p>
            <ul style="line-height: 1.6;">
                <li>🎯 勝率: 94% (vs 隨機對手)</li>
                <li>🧠 算法: PPO 強化學習</li>
                <li>🏗️ 架構: 殘差神經網絡</li>
                <li>💾 參數: 130萬+ 可訓練參數</li>
                <li>⚡ 推理: 純 NumPy 實現</li>
            </ul>
        </div>
    </div>

    <script>
        function testAPI() {
            const resultDiv = document.getElementById('result');
            resultDiv.innerHTML = '<p style="color: #1EBEFF;">🔄 測試中...</p>';
            
            fetch('/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    action: 'evaluate',
                    environment: 'connectx',
                    agents: ['submission.py', 'random'],
                    episodes: 3
                })
            })
            .then(response => response.json())
            .then(data => {
                const wins = data.filter(game => game[0] > game[1]).length;
                resultDiv.innerHTML = `
                    <div style="background: rgba(0,255,100,0.2); padding: 10px; border-radius: 5px; margin-top: 10px;">
                        ✅ 測試完成！<br>
                        🎯 結果: ${wins}/${data.length} 勝利 (${(wins/data.length*100).toFixed(1)}% 勝率)
                    </div>
                `;
            })
            .catch(error => {
                resultDiv.innerHTML = `<p style="color: #FF6B6B;">❌ 測試失敗: ${error}</p>`;
            });
        }

        // 自動檢查服務器狀態
        setInterval(() => {
            fetch('/kaggle_environments/static/player.html')
                .catch(() => {
                    document.querySelector('.status').innerHTML = 
                        '❌ 服務器連接失敗 | 🤖 AI 模型狀態未知 | 🎮 GUI 系統離線';
                    document.querySelector('.status').style.background = 'rgba(255, 107, 107, 0.2)';
                    document.querySelector('.status').style.borderColor = '#FF6B6B';
                });
        }, 30000);
    </script>
</body>
</html>
        '''

    # 添加預設遊戲數據的原版播放器
    @app.route('/kaggle_player')
    def kaggle_player_with_data():
        # 運行一場快速遊戲
        env = make("connectx", debug=False)
        
        # 載入智能體
        submission = utils.read_file("submission.py")
        ai_agent = utils.get_last_callable(submission)
        
        import random
        def random_agent(obs, config):
            valid_actions = [col for col in range(7) if obs.board[col] == 0]
            return random.choice(valid_actions)
        
        # 運行遊戲
        env.run([ai_agent, random_agent])
        
        # 獲取 ConnectX 渲染器
        connectx_js = utils.read_file("kaggle_environments/envs/connectx/connectx.js")
        
        # 生成播放器頁面
        window_kaggle = {
            "environment": {
                "name": "connectx",
                "title": "ConnectX AI Demo",
                "steps": env.steps,
                "configuration": env.configuration
            },
            "autoplay": False,  # 不自動播放，讓用戶控制
            "speed": 1000
        }
        
        return utils.get_player(window_kaggle, connectx_js)

    # 添加互動式播放器頁面
    @app.route('/player')
    def interactive_player():
        # 生成一個帶有環境選擇界面的播放器頁面
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <title>ConnectX 互動式播放器</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            backdrop-filter: blur(10px);
        }
        .control-panel {
            background: rgba(255, 255, 255, 0.1);
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            backdrop-filter: blur(10px);
        }
        .game-area {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 20px;
            min-height: 600px;
        }
        .form-group {
            margin: 15px 0;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #1EBEFF;
        }
        select, input, button {
            width: 100%;
            padding: 10px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 6px;
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-size: 14px;
        }
        button {
            background: linear-gradient(45deg, #1EBEFF, #00D4FF);
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(30, 190, 255, 0.4);
        }
        .grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
        }
        iframe {
            width: 100%;
            height: 600px;
            border: 2px solid #1EBEFF;
            border-radius: 8px;
            background: white;
        }
        .status {
            padding: 10px;
            border-radius: 6px;
            margin: 10px 0;
            font-weight: 600;
        }
        .status.success {
            background: rgba(0, 255, 100, 0.2);
            border-left: 4px solid #00FF64;
        }
        .status.error {
            background: rgba(255, 107, 107, 0.2);
            border-left: 4px solid #FF6B6B;
        }
        .status.info {
            background: rgba(30, 190, 255, 0.2);
            border-left: 4px solid #1EBEFF;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 ConnectX 互動式播放器</h1>
            <p>配置你的遊戲設置，然後觀看 AI 對戰的動態演示</p>
        </div>

        <div class="grid">
            <div class="control-panel">
                <h3>🛠️ 遊戲配置</h3>
                
                <div class="form-group">
                    <label for="environment">環境選擇:</label>
                    <select id="environment">
                        <option value="connectx">ConnectX (四子連棋)</option>
                        <option value="tictactoe">TicTacToe (井字遊戲)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="agent1">智能體 1:</label>
                    <select id="agent1">
                        <option value="submission.py">🤖 你的 AI (submission.py)</option>
                        <option value="random">🎲 隨機對手</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="agent2">智能體 2:</label>
                    <select id="agent2">
                        <option value="random">🎲 隨機對手</option>
                        <option value="submission.py">🤖 你的 AI (submission.py)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="episodes">評估局數:</label>
                    <select id="episodes">
                        <option value="1">1 局 (快速)</option>
                        <option value="3">3 局</option>
                        <option value="5">5 局</option>
                        <option value="10">10 局</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="speed">播放速度:</label>
                    <select id="speed">
                        <option value="500">快速 (0.5秒)</option>
                        <option value="1000" selected>正常 (1秒)</option>
                        <option value="2000">慢速 (2秒)</option>
                    </select>
                </div>

                <button onclick="runGame()">🚀 開始遊戲</button>
                <button onclick="runEvaluation()">📊 評估模式</button>
                
                <div id="status"></div>
                <div id="results"></div>
            </div>

            <div class="game-area">
                <h3>🎯 遊戲演示</h3>
                <iframe id="gameFrame" src="about:blank"></iframe>
            </div>
        </div>
    </div>

    <script>
        function showStatus(message, type = 'info') {
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = `<div class="status ${type}">${message}</div>`;
        }

        function runGame() {
            const env = document.getElementById('environment').value;
            const agent1 = document.getElementById('agent1').value;
            const agent2 = document.getElementById('agent2').value;
            const speed = document.getElementById('speed').value;

            showStatus('🔄 正在生成遊戲...', 'info');

            const requestData = {
                action: 'run',
                environment: env,
                agents: [agent1, agent2],
                render: { mode: 'html' }
            };

            fetch('/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            })
            .then(response => response.text())
            .then(html => {
                // 將 HTML 注入到 iframe 中
                const iframe = document.getElementById('gameFrame');
                const blob = new Blob([html], { type: 'text/html' });
                const url = URL.createObjectURL(blob);
                iframe.src = url;
                
                showStatus('✅ 遊戲已生成！使用播放控制來觀看對戰', 'success');
            })
            .catch(error => {
                console.error('Error:', error);
                showStatus('❌ 遊戲生成失敗: ' + error, 'error');
            });
        }

        function runEvaluation() {
            const env = document.getElementById('environment').value;
            const agent1 = document.getElementById('agent1').value;
            const agent2 = document.getElementById('agent2').value;
            const episodes = parseInt(document.getElementById('episodes').value);

            showStatus('📊 正在運行評估...', 'info');

            const requestData = {
                action: 'evaluate',
                environment: env,
                agents: [agent1, agent2],
                episodes: episodes
            };

            fetch('/', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                // 計算統計數據
                const agent1Wins = data.filter(game => game[0] > game[1]).length;
                const agent2Wins = data.filter(game => game[1] > game[0]).length;
                const draws = data.filter(game => game[0] === game[1]).length;

                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `
                    <div class="status success">
                        📊 評估完成！<br>
                        🥇 ${agent1}: ${agent1Wins}勝 (${(agent1Wins/episodes*100).toFixed(1)}%)<br>
                        🥈 ${agent2}: ${agent2Wins}勝 (${(agent2Wins/episodes*100).toFixed(1)}%)<br>
                        🤝 平局: ${draws}局 (${(draws/episodes*100).toFixed(1)}%)
                    </div>
                `;
                
                showStatus('✅ 評估完成！查看結果統計', 'success');
            })
            .catch(error => {
                console.error('Error:', error);
                showStatus('❌ 評估失敗: ' + error, 'error');
            });
        }

        // 初始加載一個演示遊戲
        window.onload = function() {
            showStatus('🎮 歡迎使用互動式播放器！點擊開始遊戲來體驗 AI 對戰', 'info');
            setTimeout(runGame, 1000); // 1秒後自動加載一個演示
        };
    </script>
</body>
</html>
        '''

    # 添加實時遊戲演示路由
    @app.route('/demo')
    def demo_game():
        # 運行一場 AI vs 隨機的遊戲
        env = make("connectx", debug=False)
        
        # 載入智能體
        submission = utils.read_file("submission.py")
        ai_agent = utils.get_last_callable(submission)
        
        import random
        def random_agent(obs, config):
            valid_actions = [col for col in range(7) if obs.board[col] == 0]
            return random.choice(valid_actions)
        
        # 運行遊戲
        env.run([ai_agent, random_agent])
        
        # 獲取 ConnectX 渲染器
        connectx_js = utils.read_file("kaggle_environments/envs/connectx/connectx.js")
        
        # 生成播放器頁面
        window_kaggle = {
            "environment": {
                "name": "connectx",
                "steps": env.steps,
                "configuration": env.configuration
            },
            "autoplay": True,
            "speed": 1000
        }
        
        return utils.get_player(window_kaggle, connectx_js)
    
    # 添加 API 端點用於生成新遊戲
    @app.route('/new_game')
    def new_game():
        return demo_game()
    
    # 添加主 API 路由
    @app.route('/api', methods=['GET', 'POST'])
    def api():
        return http_request(request)
    
    # 添加根路由重定向到索引頁面  
    @app.route('/', methods=['GET'])
    def root():
        # 如果是簡單的 GET 請求，重定向到索引頁面
        from flask import redirect
        return redirect('/index')
    
    # 為 POST 請求保留 API 功能
    @app.route('/', methods=['POST'])
    def root_post():
        return http_request(request)
    app.run("127.0.0.1", 8000, debug=True)


def http_request(request):
    # Set CORS headers for the preflight request
    if request.method == "OPTIONS":
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "3600",
        }

        return ("", 204, headers)

    headers = {"Access-Control-Allow-Origin": "*"}

    params = request.args.to_dict()
    for key in list(params.keys()):
        if key.endswith("[]"):
            params[key.replace("[]", "")] = request.args.getlist(key)
            del params[key]
        elif key.endswith("{}"):
            params[key.replace("{}", "")] = json.loads(params[key])
            del params[key]

    body = request.get_json(silent=True, force=True) or {}
    args = {**params, **body}
    return (action_handler(args), 200, headers)

def main():
    args = parser.parse_args()
    if args.action == "http-server":
        action_http(args)
    print(action_handler(vars(args)))

if __name__ == "__main__":
    main()
