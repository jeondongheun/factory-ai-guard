% ================================================================
% SKAB 데이터 EDA - MATLAB
% ================================================================

clear; clc; close all;

% ── 경로 설정 ──────────────────────────────────────────────────
% 본인 경로로 수정
skab_dir = '/Users/mac/Desktop/Code/factory-ai-guard/backend/ml/data/SKAB/data/';

% ── 1. 데이터 로딩 ─────────────────────────────────────────────
fprintf('=== 1. 데이터 로딩 ===\n');

% anomaly-free
af = readtable(fullfile(skab_dir, 'anomaly-free', 'anomaly-free.csv'), ...
    'Delimiter', ';', 'VariableNamingRule', 'preserve');
fprintf('anomaly-free: %d행, %d컬럼\n', height(af), width(af));

% valve1/0.csv (정상+이상 둘 다 있는 파일)
v1 = readtable(fullfile(skab_dir, 'valve1', '0.csv'), ...
    'Delimiter', ';', 'VariableNamingRule', 'preserve');
fprintf('valve1/0.csv: %d행, %d컬럼\n', height(v1), width(v1));
disp(v1(1:5, :));

% 센서 컬럼명
sensors = {'Accelerometer1RMS', 'Accelerometer2RMS', 'Current', ...
           'Pressure', 'Temperature', 'Thermocouple', ...
           'Voltage', 'Volume Flow RateRMS'};

% anomaly 분리
normal_idx  = v1.anomaly == 0;
anomaly_idx = v1.anomaly == 1;
fprintf('\n정상: %d행, 이상: %d행\n', sum(normal_idx), sum(anomaly_idx));

% ── 2. 센서별 정상 vs 이상 평균 비교 ───────────────────────────
fprintf('\n=== 2. 센서별 평균 비교 ===\n');
fprintf('%-25s %10s %10s %10s\n', '센서', '정상평균', '이상평균', '차이');
fprintf('%s\n', repmat('-', 1, 55));

for i = 1:length(sensors)
    col = sensors{i};
    if ismember(col, v1.Properties.VariableNames)
        n_mean = mean(v1.(col)(normal_idx),  'omitnan');
        a_mean = mean(v1.(col)(anomaly_idx), 'omitnan');
        fprintf('  %-23s %10.3f %10.3f %10.3f\n', col, n_mean, a_mean, abs(a_mean - n_mean));
    end
end

% ── 3. 센서별 분포 히스토그램 ──────────────────────────────────
figure('Name', '센서별 분포', 'Position', [100 100 1400 900]);
sgtitle('센서별 정상 vs 이상 분포 (valve1/0.csv)', 'FontSize', 14, 'FontWeight', 'bold');

for i = 1:length(sensors)
    col = sensors{i};
    subplot(4, 2, i);
    if ismember(col, v1.Properties.VariableNames)
        hold on;
        histogram(v1.(col)(normal_idx),  50, 'FaceColor', '#3498db', ...
                  'FaceAlpha', 0.6, 'DisplayName', '정상');
        histogram(v1.(col)(anomaly_idx), 50, 'FaceColor', '#e74c3c', ...
                  'FaceAlpha', 0.6, 'DisplayName', '이상');
        title(col, 'FontSize', 9);
        legend('FontSize', 7);
        grid on; hold off;
    end
end

% ── 4. 시계열 + 이상 구간 표시 ─────────────────────────────────
figure('Name', '시계열', 'Position', [100 100 1600 900]);
sgtitle('센서 시계열 + 이상 구간 (valve1/0.csv)', 'FontSize', 14, 'FontWeight', 'bold');

t = 1:height(v1);
anom = v1.anomaly;

for i = 1:length(sensors)
    col = sensors{i};
    subplot(4, 2, i);
    if ismember(col, v1.Properties.VariableNames)
        hold on;
        plot(t, v1.(col), 'Color', '#3498db', 'LineWidth', 0.8);

        % 이상 구간 빨간 음영
        in_anom = false;
        for j = 1:length(anom)
            if anom(j) == 1 && ~in_anom
                x_start = j; in_anom = true;
            elseif anom(j) == 0 && in_anom
                xregion(x_start, j, 'FaceColor', 'red', 'FaceAlpha', 0.2, ...
                        'EdgeColor', 'none');
                in_anom = false;
            end
        end
        if in_anom
            xregion(x_start, length(anom), 'FaceColor', 'red', 'FaceAlpha', 0.2, ...
                    'EdgeColor', 'none');
        end

        title(col, 'FontSize', 9);
        grid on; hold off;
    end
end

% ── 5. 상관관계 히트맵 ─────────────────────────────────────────
figure('Name', '상관관계', 'Position', [100 100 900 800]);

avail = {};
for i = 1:length(sensors)
    if ismember(sensors{i}, v1.Properties.VariableNames)
        avail{end+1} = sensors{i};
    end
end

sensor_data = zeros(height(v1), length(avail));
for i = 1:length(avail)
    sensor_data(:, i) = v1.(avail{i});
end

corr_mat = corr(sensor_data, 'rows', 'complete');

% redblue 대신 imagesc + colormap 직접 사용
imagesc(corr_mat);

% colormap 커스텀 (파랑-하양-빨강)
n = 256;
r = [linspace(0, 1, n/2), ones(1, n/2)];
g = [linspace(0, 1, n/2), linspace(1, 0, n/2)];
b = [ones(1, n/2), linspace(1, 0, n/2)];
colormap([r' g' b']);
colorbar;
caxis([-1 1]);

% 축 라벨
set(gca, 'XTick', 1:length(avail), 'XTickLabel', avail, ...
         'YTick', 1:length(avail), 'YTickLabel', avail, ...
         'XTickLabelRotation', 45, 'FontSize', 8);
title('센서간 상관관계', 'FontSize', 13);

% 수치 표시
for i = 1:length(avail)
    for j = 1:length(avail)
        text(j, i, sprintf('%.2f', corr_mat(i,j)), ...
             'HorizontalAlignment', 'center', 'FontSize', 8, ...
             'Color', 'black');
    end
end

% ── 6. 이상 시작 전후 패턴 ─────────────────────────────────────
figure('Name', '이상 전후 패턴', 'Position', [100 100 1400 900]);
sgtitle('이상 시작 전후 패턴 (±30 timestep)', 'FontSize', 14, 'FontWeight', 'bold');

% 첫 번째 이상 시작 지점 찾기
anom_starts = find(diff([0; anom]) == 1);
fprintf('\n이상 시작 지점: ');
disp(anom_starts');

if ~isempty(anom_starts)
    s = anom_starts(1);
    s_start = max(1, s - 30);
    s_end   = min(height(v1), s + 60);
    window  = v1(s_start:s_end, :);
    x_range = 1:height(window);
    ref_line = s - s_start + 1;  % 이상 시작 위치

    for i = 1:length(sensors)
        col = sensors{i};
        subplot(4, 2, i);
        if ismember(col, window.Properties.VariableNames)
            hold on;
            plot(x_range, window.(col), 'Color', '#3498db', 'LineWidth', 1.2);
            xline(ref_line, 'k--', '이상시작', 'LineWidth', 1.5);

            % 이상 구간 음영
            w_anom = window.anomaly;
            in_anom = false;
            for j = 1:length(w_anom)
                if w_anom(j) == 1 && ~in_anom
                    x_start = j; in_anom = true;
                elseif w_anom(j) == 0 && in_anom
                    xregion(x_start, j, 'FaceColor', 'red', 'FaceAlpha', 0.25, ...
                            'EdgeColor', 'none');
                    in_anom = false;
                end
            end
            if in_anom
                xregion(x_start, length(w_anom), 'FaceColor', 'red', ...
                        'FaceAlpha', 0.25, 'EdgeColor', 'none');
            end

            title(col, 'FontSize', 9);
            grid on; hold off;
        end
    end
end

fprintf('\n✅ EDA 완료!\n');