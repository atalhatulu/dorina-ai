package com.dorina.assistant;

import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.camera2.CameraManager;
import android.media.AudioManager;
import android.net.Uri;
import android.os.BatteryManager;
import android.os.Build;
import android.provider.AlarmClock;
import android.provider.Settings;

import org.json.JSONObject;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class AndroidDeviceActions {

    private Context context;

    public AndroidDeviceActions(Context context) {
        this.context = context;
    }

    public String executeAction(String toolName, JSONObject args) {
        try {
            switch (toolName) {
                case "get_battery_status":
                    return getBatteryStatus();

                case "set_brightness":
                    int level = args.optInt("level", 100);
                    return setBrightness(level);

                case "set_volume":
                    int vol = args.optInt("level", 50);
                    return setVolume(vol);

                case "toggle_flashlight":
                    boolean enabled = args.optBoolean("enabled", true);
                    return toggleFlashlight(enabled);

                case "open_app":
                    String appName = args.optString("app_name", "");
                    return openApp(appName);

                case "set_alarm":
                    String time = args.optString("time", "08:00");
                    String label = args.optString("label", "Alarm");
                    return setAlarm(time, label);

                default:
                    return "Bilinmeyen eylem: " + toolName;
            }
        } catch (Exception e) {
            return "Hata oluştu: " + e.getMessage();
        }
    }

    public String getBatteryStatus() {
        BatteryManager bm = (BatteryManager) context.getSystemService(Context.BATTERY_SERVICE);
        if (bm != null) {
            int batLevel = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY);
            return "🔋 Telefonunun bataryası şu anda %" + batLevel + " seviyesinde Talha!";
        }
        return "Batarya bilgisi alınamadı.";
    }

    public String setBrightness(int percentage) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            if (!Settings.System.canWrite(context)) {
                Intent intent = new Intent(Settings.ACTION_MANAGE_WRITE_SETTINGS);
                intent.setData(Uri.parse("package:" + context.getPackageName()));
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                context.startActivity(intent);
                return "⚠️ Ekran parlaklığını doğrudan kontrol edebilmem için lütfen açılan ekrandan sistem ayarlarını değiştirme iznini aç!";
            }
        }

        try {
            Settings.System.putInt(context.getContentResolver(),
                    Settings.System.SCREEN_BRIGHTNESS_MODE,
                    Settings.System.SCREEN_BRIGHTNESS_MODE_MANUAL);

            int val = (int) (percentage * 255.0f / 100.0f);
            if (val < 1) val = 1;
            if (val > 255) val = 255;

            Settings.System.putInt(context.getContentResolver(),
                    Settings.System.SCREEN_BRIGHTNESS, val);

            return "☀️ Ekran parlaklığı %" + percentage + " seviyesine ayarlandı!";
        } catch (Exception e) {
            return "Parlaklık ayarlanamadı: " + e.getMessage();
        }
    }

    public String setVolume(int percentage) {
        AudioManager am = (AudioManager) context.getSystemService(Context.AUDIO_SERVICE);
        if (am != null) {
            int maxVol = am.getStreamMaxVolume(AudioManager.STREAM_MUSIC);
            int targetVol = (int) (percentage * maxVol / 100.0f);
            am.setStreamVolume(AudioManager.STREAM_MUSIC, targetVol, AudioManager.FLAG_SHOW_UI);
            return "🔊 Telefon medya sesi %" + percentage + " seviyesine ayarlandı!";
        }
        return "Ses ayarlanamadı.";
    }

    public String toggleFlashlight(boolean enabled) {
        CameraManager cm = (CameraManager) context.getSystemService(Context.CAMERA_SERVICE);
        if (cm != null) {
            try {
                String cameraId = cm.getCameraIdList()[0];
                cm.setTorchMode(cameraId, enabled);
                return enabled ? "🔦 El feneri açıldı!" : "🔦 El feneri kapatıldı!";
            } catch (Exception e) {
                return "Fener kontrol hatası: " + e.getMessage();
            }
        }
        return "Fener donanımı bulunamadı.";
    }

    public String setAlarm(String time, String label) {
        try {
            int hour = 8;
            int min = 0;

            Matcher m = Pattern.compile("(\\d{1,2})[:.](\\d{2})").matcher(time);
            if (m.find()) {
                hour = Integer.parseInt(m.group(1));
                min = Integer.parseInt(m.group(2));
            } else {
                Matcher m2 = Pattern.compile("(\\d{1,2})").matcher(time);
                if (m2.find()) {
                    hour = Integer.parseInt(m2.group(1));
                }
            }

            Intent intent = new Intent(AlarmClock.ACTION_SET_ALARM);
            intent.putExtra(AlarmClock.EXTRA_HOUR, hour);
            intent.putExtra(AlarmClock.EXTRA_MINUTES, min);
            intent.putExtra(AlarmClock.EXTRA_MESSAGE, label);
            // DO NOT set EXTRA_SKIP_UI to avoid Samsung OneUI Permission Denial
            intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);

            context.startActivity(intent);
            String fmt = String.format("%02d:%02d", hour, min);
            return "⏰ Saat " + fmt + " için '" + label + "' alarmı başarıyla kuruldu!";
        } catch (Exception e) {
            return "Alarm kurulamadı: " + e.getMessage();
        }
    }

    public String openApp(String rawAppName) {
        String norm = TurkishNormalizer.normalize(rawAppName);
        Map<String, String> packageMap = new HashMap<>();
        packageMap.put("whatsapp", "com.whatsapp");
        packageMap.put("vatsap", "com.whatsapp");
        packageMap.put("youtube", "com.google.android.youtube");
        packageMap.put("tiktok", "com.zhiliaoapp.musically");
        packageMap.put("instagram", "com.instagram.android");
        packageMap.put("spotify", "com.spotify.music");
        packageMap.put("chatgpt", "com.openai.chatgpt");
        packageMap.put("kamera", "com.sec.android.app.camera");
        packageMap.put("galeri", "com.sec.android.gallery3d");
        packageMap.put("notlar", "com.samsung.android.app.notes");
        packageMap.put("ziraat", "com.ziraat.ziraatmobil");
        packageMap.put("trendyol", "trendyol.com");

        String targetPkg = null;
        for (String k : packageMap.keySet()) {
            if (norm.contains(k)) {
                targetPkg = packageMap.get(k);
                break;
            }
        }

        PackageManager pm = context.getPackageManager();
        if (targetPkg != null) {
            Intent intent = pm.getLaunchIntentForPackage(targetPkg);
            if (intent != null) {
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                context.startActivity(intent);
                return "▶️ " + rawAppName + " uygulaması açılıyor!";
            }
        }

        // Direct package name search fallback
        try {
            Intent intent = pm.getLaunchIntentForPackage("com." + norm);
            if (intent != null) {
                intent.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                context.startActivity(intent);
                return "▶️ " + rawAppName + " açıldı!";
            }
        } catch (Exception ignored) {}

        return "📱 " + rawAppName + " uygulaması telefonda bulunamadı.";
    }
}
