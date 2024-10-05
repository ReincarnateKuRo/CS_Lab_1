# 
## 一般狀況(從專案創建到自己推送自己的修改到github)
- 先在自己的github建立repository
- 打開cmd依序輸入以下指令
	- cd導航到專案目錄
	```
	cd C:\path\to\your\project
	如果是D槽：
	D:
	cd D:\path\to\your\project
	```
	- 初始化
	```
	git init
	```
	- 設定遠端的repository
	```
	git remote add origin https://github.com/你的帳號/cs_lab.git
	```
	- 提交專案到本地git
	```
	git add .
	```
	```
	git commit -m "初始提交"
	```
	- for我的電腦的git，預設傳送分支是master而不是main這檔事，將分支重新命名
	```
	git branch -m master main
	```
	- 將專案推送到遠端repository (github)
	```
	git push -u origin main
	```

## 一般狀況(協作者上傳修改到他自己的分支上時)(整合分支到主分支)
- 到那個專案的repository點擊`Pull request`後再點擊`New Pull request`
- 在右側的分支選項中，選擇你同學 push 上來的那個分支作為 **source branch**，並選擇你的主分支（如 `main`）作為 **target branch**
- 點擊 `Create pull request`
-  進入 PR 頁面，檢查代碼的變更內容。
- 如果確認沒有問題，點擊 `Merge pull request` 按鈕，並選擇 "Confirm merge"。
- 一旦 PR 合併完成，該分支的修改就會合併到主分支上了
- 之後要更新本地端的主分支`main`，打開cmd輸入以下指令
	- cd導航到專案目錄(如果已經在該專案目錄可跳過)
	- 切換分支到main(如果分支已經在main可以跳過)
	- 拉取最新的更新
	```
	git pull origin main
	```
## 想要轉換主分支(專案擁有者)
- 在此以主分支從master分支轉換到main為例
- 打開cmd依序輸入以下指令
	- 改設定預設分支為main
	```
	git branch -M main
	```
	- 將專案推送到遠端repository(如果這步驟成功請直接跳到刪除本地master分支環節)
	```
	git pull origin main --rebase
	```
	- 如果推送後發生以下錯誤代表遠端的 `main` 分支已經有一些內容，而你的本地 `main` 分支沒有包含這些內容，因此推送被拒絕了。你需要將遠端的變更合併到本地分支後再進行推送(做完這個步驟後先回到上一個步驟再推送一次，如果失敗則進到下一個步驟)![[Pasted image 20241005182600.png]]
	```
	git pull origin main --rebase
	```
	- 如果你確定本地的變更應該覆蓋遠端的內容，並不想保留遠端的變更，你可以使用強制推送
	```
	git push -f origin main
	```
	- (刪除本地master分支環節)請先確保沒有裝置在使用master分支，如果你的裝置在使用master分支請輸入以下指令，如果沒有則跳到下一個步驟
	```
	git checkout 不是master的branch
	```
	- 刪除本地master分支，**如果顯示以下錯誤請用第二條指令刪除**![[Pasted image 20241005182702.png]]
	```
	git branch -d master
	git branch -D master
	```
	- 刪除遠端github master 分支
	```
	git push origin --delete master
	```
	
	
# 專案協作者
## 去接受協作邀請
- 協作者在github上收到專案負責人的通知協作Notification
- 去註冊github的電郵找邀請並同意
- 回到github dashboard點選左上角的`三條線`可以看到協作的repository
## cmd要幹嘛(從clone專案到push自己的改動)
- 打開cmd依序輸入以下指令
	- clone專案負責人的repository
	```
	git clone https://github.com/username/repository.git
	```
	- cd導航到專案目錄
	```
	cd C:\path\to\your\project
	如果是D槽：
	D:
	cd D:\path\to\your\project
	```
	- 創建自己的作業分支(如果已經創建過那用第二行)
	```
	git checkout -b your_branch_name
	git checkout your_branch_name
	```
	- 確認分支在哪，確認完後## 跳到spyder ide要幹嘛
	```
	git branch
	```
	- (編輯好後)提交專案到本地以及推送專案到遠端
	```
	git add .
	```
	```
	git commit -m "Your commit message"
	```
	```
	git push origin your_branch_name
	```
## cmd要幹嘛(專案負責人整合好之後要進行下一次的修改時)
- 打開cmd依序輸入以下指令
	- 添加你的 repository 作為 `upstream` (如果不是第一次在專案負責人整合後同步專案可以跳果這個步驟)
	```
	git remote add upstream https://github.com/你的帳號/專案名稱.git
	```
	- 確認是否添加成功(理論上會得到一個origin和一個upstream)
	```
	git remote -v
	```
	- 拉取最新的變更
	```
	git fetch upstream
	```
	```
	git merge upstream/main
	```
	- 確保本地端為最新同步
	```
	git checkout main
	```
	```
	git pull upstream main
	```
## spyder ide要幹嘛
- 打開要作業的spyder
	- 在spyder內安裝好專案需要的模組包
	- 點Project然後按open project ![[Pasted image 20241005175440.png]]
	- 然後尋找到你要打開的專案，點選後按開啟
	- 修改完檔案或儲存好檔案後跳到# cmd要幹嘛 的(編輯好後步驟)